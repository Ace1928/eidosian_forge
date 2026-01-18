from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import step_fn
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def single_loss_example(optimizer_fn, distribution, use_bias=False, iterations_per_step=1):
    """Build a very simple network to use in tests and examples."""

    def dataset_fn():
        return dataset_ops.Dataset.from_tensors([[1.0]]).repeat()
    optimizer = optimizer_fn()
    layer = core.Dense(1, use_bias=use_bias)

    def loss_fn(ctx, x):
        del ctx
        y = array_ops.reshape(layer(x), []) - constant_op.constant(1.0)
        return y * y
    single_loss_step = step_fn.StandardSingleLossStep(dataset_fn, loss_fn, optimizer, distribution, iterations_per_step)
    return (single_loss_step, layer)