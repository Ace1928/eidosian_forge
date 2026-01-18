from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.models import Functional
from keras.src.ops import operation_utils
from keras.src.utils import file_utils
def stack_residual_blocks_v1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
        x: Input tensor.
        filters: Number of filters in the bottleneck layer in a block.
        blocks: Number of blocks in the stacked blocks.
        stride1: Stride of the first layer in the first block. Defaults to `2`.
        name: Stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = residual_block_v1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = residual_block_v1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x