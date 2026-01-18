import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
def run_distribute_coordinator(worker_fn, strategy, eval_fn=None, eval_strategy=None, cluster_spec=None, task_type=None, task_id=None, session_config=None, rpc_layer='grpc'):
    """Runs the coordinator for distributed TensorFlow.

  This function runs a split coordinator for distributed TensorFlow in its
  default mode, i.e the STANDALONE_CLIENT mode. Given a `cluster_spec`
  specifying server addresses and their roles in a cluster, this coordinator
  will figure out how to set them up, give the underlying function the right
  targets for master sessions via a scope object and coordinate their training.
  The cluster consisting of standard servers needs to be brought up either with
  the standard server binary or with a binary running distribute coordinator
  with `task_type` set to non-client type which will then turn into standard
  servers.

  In addition to be the distribute coordinator, this is also the source of
  configurations for each job in the distributed training. As there are multiple
  ways to configure a distributed TensorFlow cluster, its context object
  provides these configurations so that users or higher-level APIs don't have to
  figure out the configuration for each job by themselves.

  In the between-graph replicated training, this coordinator will create
  multiple threads and each calls the `worker_fn` which is supposed to create
  its own graph and connect to one worker master given by its context object. In
  the in-graph replicated training, it has only one thread calling this
  `worker_fn`.

  Another mode is the INDEPENDENT_WORKER mode where each server runs a
  distribute coordinator which will start a standard server and optionally runs
  `worker_fn` depending whether it is between-graph training or in-graph
  replicated training.

  The `strategy` object is expected to be a DistributionStrategy object which
  has implemented methods needed by distributed coordinator such as
  `configure(session_config, cluster_spec, task_type, task_id)` which configures
  the strategy object for a specific task and `experimental_should_init`
  property which instructs the distribute coordinator whether to run init ops
  for a task. The distribute coordinator will make a copy of the `strategy`
  object, call its `configure` method and pass it to `worker_fn` as an argument.

  The `worker_fn` defines the training logic and is called under its own
  worker context which can be accessed to via `get_current_worker_context`. A
  worker context provides access to configurations for each task, e.g. the
  task_type, task_id, master target and so on. Since `worker_fn` will be called
  in a thread and possibly multiple times, caller should be careful when it
  accesses global data. For example, it is unsafe to define flags in a
  `worker_fn` or to define different environment variables for different
  `worker_fn`s.

  The `worker_fn` for the between-graph replication is defined as if there is
  only one worker corresponding to the `worker_fn` and possibly ps jobs. For
  example, when training with parameter servers, it assigns variables to
  parameter servers and all other operations to that worker. In the in-graph
  replication case, the `worker_fn` has to define operations for all worker
  jobs. Using a distribution strategy can simplify the `worker_fn` by not having
  to worry about the replication and device assignment of variables and
  operations.

  This method is intended to be invoked by high-level APIs so that users don't
  have to explicitly call it to run this coordinator. For those who don't use
  high-level APIs, to change a program to use this coordinator, wrap everything
  in a the program after global data definitions such as commandline flag
  definition into the `worker_fn` and get task-specific configurations from
  the worker context.

  The `cluster_spec` can be either passed by the argument or parsed from the
  "TF_CONFIG" environment variable. Example of a TF_CONFIG:
  ```
    cluster = {'chief': ['host0:2222'],
               'ps': ['host1:2222', 'host2:2222'],
               'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
    os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster})
  ```

  If `cluster_spec` is not given in any format, it becomes local training and
  this coordinator will connect to a local session.

  For evaluation, if "evaluator" exists in the cluster_spec, a separate thread
  will be created to call `eval_fn` with its `task_type` set to "evaluator". If
  `eval_fn` is not defined, fall back to `worker_fn`. This implies that
  evaluation will be done on a single machine if there is an "evaluator" task.
  If "evaluator" doesn't exist in the cluster_spec, it entirely depends on the
  `worker_fn` for how to do evaluation.

  Args:
    worker_fn: the function to be called. The function should accept a
      `strategy` object and will be given access to a context object via a
      context manager scope.
    strategy: a DistributionStrategy object specifying whether it should run
      between-graph replicated training or not, whether to run init ops, etc.
      This object will also be configured given `session_config`,
      `cluster_spec`, `task_type` and `task_id`.
    eval_fn: optional function for "evaluator" task. If `eval_fn` is not passed
      in but a "evaluator" task is found in the `cluster_spec`, the `worker_fn`
      will be used for this task.
    eval_strategy: optional DistributionStrategy object for "evaluator" task.
    cluster_spec: a dict, ClusterDef or ClusterSpec specifying servers and roles
      in a cluster. If not set or empty, fall back to local training.
    task_type: the current task type, optional if this is a client.
    task_id: the current task id, optional if this is a client.
    session_config: an optional `tf.compat.v1.ConfigProto` object which will be
      passed to `strategy`'s `configure` method and used to create a session.
    rpc_layer: optional string, the protocol for RPC, e.g. "grpc".

  Raises:
    ValueError: if `cluster_spec` is supplied but not a dict or a ClusterDef or
      a ClusterSpec.

  Returns:
    In the client job, return the value returned by `worker_fn` if
    it is in-graph replication or INDEPENDENT_WORKER mode; return None
    otherwise.
  """
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    rpc_layer = tf_config.get('rpc_layer', rpc_layer)
    environment = tf_config.get('environment', None)
    if not cluster_spec:
        cluster_spec = tf_config.get('cluster', {})
        task_env = tf_config.get('task', {})
        if task_env:
            task_type = task_env.get('type', task_type)
            task_id = int(task_env.get('index', task_id))
    if cluster_spec:
        cluster_spec = normalize_cluster_spec(cluster_spec)
    elif hasattr(strategy.extended, '_cluster_resolver'):
        cluster_resolver = strategy.extended._cluster_resolver
        task_type = cluster_resolver.task_type
        task_id = cluster_resolver.task_id
        rpc_layer = cluster_resolver.rpc_layer or rpc_layer
        environment = cluster_resolver.environment
        cluster_spec = cluster_resolver.cluster_spec()
    session_config = session_config or config_pb2.ConfigProto(allow_soft_placement=True)
    if cluster_spec:
        logging.info('Running Distribute Coordinator with cluster_spec = %r, task_type = %r, task_id = %r, environment = %r, rpc_layer = %r', cluster_spec.as_dict(), task_type, task_id, environment, rpc_layer)
    if not cluster_spec:
        logging.info('Running local Distribute Coordinator.')
        _run_single_worker(worker_fn, strategy, None, None, None, session_config, rpc_layer)
        if eval_fn:
            _run_single_worker(eval_fn, eval_strategy, None, None, None, session_config, rpc_layer)
        else:
            logging.warning('Skipped evaluation since `eval_fn` is not passed in.')
    else:
        if not eval_fn:
            logging.warning('`eval_fn` is not passed in. The `worker_fn` will be used if an "evaluator" task exists in the cluster.')
        eval_fn = eval_fn or worker_fn
        if not eval_strategy:
            logging.warning('`eval_strategy` is not passed in. No distribution strategy will be used for evaluation.')
        _configure_session_config_for_std_servers(strategy, eval_strategy, session_config, cluster_spec, task_type, task_id)
        if task_type != _TaskType.EVALUATOR and (not getattr(strategy.extended, '_std_server_started', False)):
            server = _run_std_server(cluster_spec=cluster_spec, task_type=task_type, task_id=task_id, session_config=session_config, rpc_layer=rpc_layer, environment=environment)
        if task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
            if strategy.extended.experimental_between_graph:
                return _run_single_worker(worker_fn, strategy, cluster_spec, task_type, task_id, session_config, rpc_layer)
            else:
                context = _WorkerContext(strategy, cluster_spec, task_type, task_id)
                if context.is_chief:
                    return _run_single_worker(worker_fn, strategy, cluster_spec, None, None, session_config, rpc_layer)
                else:
                    server.join()
        elif task_type == _TaskType.EVALUATOR:
            return _run_single_worker(eval_fn, eval_strategy, cluster_spec, task_type, task_id, session_config, rpc_layer)
        else:
            if task_type != _TaskType.PS:
                raise ValueError('Unexpected task_type: %r' % task_type)
            server.join()