import copy
import math
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def block(inputs, activation='swish', drop_rate=0.0, name='', filters_in=32, filters_out=16, kernel_size=3, strides=1, expand_ratio=1, se_ratio=0.0, id_skip=True):
    """An inverted residual block.

    Args:
        inputs: input tensor.
        activation: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.

    Returns:
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation, name=name + 'expand_activation')(x)
    else:
        x = inputs
    if strides == 2:
        x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, kernel_size), name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=conv_pad, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER, name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation, name=name + 'activation')(x)
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = layers.Reshape(se_shape, name=name + 'se_reshape')(se)
        se = layers.Conv2D(filters_se, 1, padding='same', activation=activation, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'se_reduce')(se)
        se = layers.Conv2D(filters, 1, padding='same', activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')
    x = layers.Conv2D(filters_out, 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if id_skip and strides == 1 and (filters_in == filters_out):
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')
    return x