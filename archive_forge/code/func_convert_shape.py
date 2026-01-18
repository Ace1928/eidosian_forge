import inspect
import itertools
import string
from absl import logging
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.layers import Layer
from keras.src.models import Functional
from keras.src.models import Sequential
from keras.src.utils import io_utils
from keras.src.utils import tree
from keras.src.utils.module_utils import tensorflow as tf
def convert_shape(x):
    poly_shape = []
    for index, dim in enumerate(list(x.shape)):
        if dim is not None:
            poly_shape.append(str(dim))
        elif not allow_none:
            raise ValueError(f'Illegal None dimension in {x} with shape {x.shape}')
        elif index == 0:
            poly_shape.append('batch')
        else:
            poly_shape.append(next(dim_names))
    return '(' + ', '.join(poly_shape) + ')'