import copy
import six
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
def estimator_evaluate(estimator, evaluate_distributed_fn, hooks):
    """Run distribute coordinator for Estimator's `evaluate` method."""
    assert estimator._config._distribute_coordinator_mode
    run_config = estimator._config
    assert estimator._config.cluster_spec
    cluster_spec = multi_worker_util.normalize_cluster_spec(estimator._config.cluster_spec)
    assert estimator._config._eval_distribute
    if 'evaluator' in cluster_spec.jobs:
        raise ValueError("'evaluator' job is not supported if you don't use `train_and_evaluate`")
    if estimator._config._distribute_coordinator_mode != dc.CoordinatorMode.STANDALONE_CLIENT:
        raise ValueError('Only `STANDALONE_CLIENT` mode is supported when you call `Estimator.evaluate`')
    if estimator._config._eval_distribute.extended.experimental_between_graph:
        raise ValueError('`Estimator.evaluate` API is not supported for %s with `STANDALONE_CLIENT` mode.' % estimator._config._eval_distribute.__class__.__name__)

    def _worker_fn(strategy):
        """Function for evaluation."""
        local_estimator = copy.deepcopy(estimator)
        local_estimator._config._eval_distribute = strategy
        context = dc_context.get_current_worker_context()
        _init_run_config_from_worker_context(local_estimator._config, context)
        logging.info('Updated config: %s', str(vars(local_estimator._config)))
        local_estimator._eval_distribution = strategy
        if context.is_chief:
            chief_hooks = hooks
        else:
            chief_hooks = []
        return evaluate_distributed_fn(local_estimator, strategy, chief_hooks)
    return dc.run_distribute_coordinator(_worker_fn, estimator._config.eval_distribute, mode=run_config._distribute_coordinator_mode, cluster_spec=cluster_spec, session_config=run_config.session_config)