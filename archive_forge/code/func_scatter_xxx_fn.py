import contextlib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication
def scatter_xxx_fn(var, sparse_delta, use_locking=False, name=None):
    del use_locking
    handle = var.handle
    with _maybe_enter_graph(handle), _maybe_on_device(var):
        op = raw_scatter_xxx_fn(handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, var.dtype), name=name)
        with ops.control_dependencies([op]):
            return var._read_variable_op()