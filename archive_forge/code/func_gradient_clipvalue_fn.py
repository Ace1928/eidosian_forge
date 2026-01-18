from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.ops import clip_ops
from tensorflow.python.platform import tf_logging as logging
def gradient_clipvalue_fn(grads_and_vars):
    if isinstance(distribute_lib.get_strategy(), (central_storage_strategy.CentralStorageStrategy, central_storage_strategy.CentralStorageStrategyV1)):
        raise ValueError('`clipvalue` is not supported with `CenteralStorageStrategy`')
    clipped_grads_and_vars = [(clip_ops.clip_by_value(g, -clipvalue, clipvalue), v) for g, v in grads_and_vars]
    return clipped_grads_and_vars