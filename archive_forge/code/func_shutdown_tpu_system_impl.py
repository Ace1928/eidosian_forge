import gc
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.tpu import tpu
from tensorflow.python.util import compat
def shutdown_tpu_system_impl(cluster_resolver, tpu_cluster_resolver_cls):
    """Implementation for tpu.experimental.shutdown_tpu_system.

  Kept separate to avoid tpu_oss code duplication.

  Shuts down the TPU devices.

  This will clear all caches, even those that are maintained through sequential
  calls to tf.tpu.experimental.initialize_tpu_system, such as the compilation
  cache.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
    tpu_cluster_resolver_cls: a reference to
        tf.distribute.cluster_resolver.TPUClusterResolver so that an instance
        of it can be initialized if cluster_resolver is None.

  Raises:
    RuntimeError: If no TPU devices found for eager execution or if run in a
        tf.function.
    TypeError: If tpu_cluster_resolver_cls is
        not tf.distribute.cluster_resolver.TPUClusterResolver.
  """
    if tpu_cluster_resolver_cls is None or not issubclass(tpu_cluster_resolver_cls, cluster_resolver_lib.ClusterResolver) or (not hasattr(tpu_cluster_resolver_cls, 'tpu_hardware_feature')):
        raise TypeError('tpu_cluster_resolver_cls is not tf.distribute.cluster_resolver.TPUClusterResolver.')
    job = None
    if cluster_resolver is None:
        if context.executing_eagerly():
            curr_device = device.DeviceSpec.from_string(context.context().device_name)
            if curr_device.job is not None:
                job = '{}/replica:0/task:0'.format(curr_device.job)
        cluster_resolver = tpu_cluster_resolver_cls('')
    assert isinstance(cluster_resolver, tpu_cluster_resolver_cls)
    tpu_name = compat.as_text(cluster_resolver._tpu)
    if tpu_name not in _INITIALIZED_TPU_SYSTEMS:
        logging.warning('You are shutting down a TPU system %s that has not been initialized.' % tpu_name)
    logging.info('Shutting down the TPU system: %s', tpu_name)
    if context.executing_eagerly():
        if tpu_name not in _LOCAL_MASTERS:
            job = '{}/replica:0/task:0'.format(cluster_resolver.get_job_name())

        @def_function.function(autograph=False)
        def _tpu_shutdown_fn():
            tpu.shutdown_system(job=job)
        run_eagerly = def_function.functions_run_eagerly()
        if run_eagerly:
            logging.warning('It looks like tf.function behavior was disabled, perhaps using tf.config.run_functions_eagerly. tf.tpu.experimental.shutdown_tpu_system requires tf.function to work. This primitive will override the disable.')
            def_function.run_functions_eagerly(False)
        try:
            with ops.device(tpu._tpu_system_device_name(job)):
                _tpu_shutdown_fn()
        finally:
            if run_eagerly is not None:
                def_function.run_functions_eagerly(run_eagerly)
        logging.info('Clearing out eager caches')
        context.context()._clear_caches()
        context.context().clear_kernel_cache()
    elif not ops.executing_eagerly_outside_functions():
        master = cluster_resolver.master()
        cluster_spec = cluster_resolver.cluster_spec()
        session_config = config_pb2.ConfigProto(allow_soft_placement=True)
        if cluster_spec:
            session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
        with ops.Graph().as_default():
            with session_lib.Session(config=session_config, target=master) as sess:
                sess.run(tpu.shutdown_system())
    else:
        raise RuntimeError('initialize_tpu_system is not supported within tf.functions.  You should call initialize_tpu_system outside of your tf.function. ')
    logging.info('Finished shutting down TPU system.')
    if tpu_name in _INITIALIZED_TPU_SYSTEMS:
        del _INITIALIZED_TPU_SYSTEMS[tpu_name]