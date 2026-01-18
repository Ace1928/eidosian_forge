import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src import backend
from keras.src import layers as keras_layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds an Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
    - Inception-ResNet-A: `block_type='block35'`
    - Inception-ResNet-B: `block_type='block17'`
    - Inception-ResNet-C: `block_type='block8'`

    Args:
      x: input tensor.
      scale: scaling factor to scale the residuals (i.e., the output of passing
        `x` through an inception module) before adding them to the shortcut
        branch. Let `r` be the output from the residual branch, the output of
        this block will be `x + scale * r`.
      block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
        structure in the residual branch.
      block_idx: an `int` used for generating layer names. The Inception-ResNet
        blocks are repeated many times in this network. We use `block_idx` to
        identify each of the repetitions. For example, the first
        Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
        and the layer names will have a common prefix `'block35_0'`.
      activation: activation function to use at the end of the block (see
        [activations](../activations.md)). When `activation=None`, no activation
        is applied
        (i.e., "linear" activation: `a(x) = x`).

    Returns:
        Output tensor for the block.

    Raises:
      ValueError: if `block_type` is not one of `'block35'`,
        `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. Expects "block35", "block17" or "block8", but got: ' + str(block_type))
    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed, backend.int_shape(x)[channel_axis], 1, activation=None, use_bias=True, name=block_name + '_conv')
    x = CustomScaleLayer(scale)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x