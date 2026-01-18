import datetime
import io
import json
import tempfile
import warnings
import zipfile
import ml_dtypes
import numpy as np
from keras.src import backend
from keras.src.backend.common import global_state
from keras.src.layers.layer import Layer
from keras.src.losses.loss import Loss
from keras.src.metrics.metric import Metric
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.utils import file_utils
from keras.src.utils import naming
from keras.src.version import __version__ as keras_version
def get_attr_skiplist(obj_type):
    skiplist = global_state.get_global_attribute(f'saving_attr_skiplist_{obj_type}', None)
    if skiplist is not None:
        return skiplist
    skiplist = ['_self_unconditional_dependency_names']
    if obj_type == 'Layer':
        ref_obj = Layer()
        skiplist += dir(ref_obj)
    elif obj_type == 'Functional':
        ref_obj = Layer()
        skiplist += dir(ref_obj) + ['operations', '_operations']
    elif obj_type == 'Sequential':
        ref_obj = Layer()
        skiplist += dir(ref_obj) + ['_functional']
    elif obj_type == 'Metric':
        ref_obj_a = Metric()
        ref_obj_b = CompileMetrics([], [])
        skiplist += dir(ref_obj_a) + dir(ref_obj_b)
    elif obj_type == 'Optimizer':
        ref_obj = Optimizer(1.0)
        skiplist += dir(ref_obj)
        skiplist.remove('variables')
    elif obj_type == 'Loss':
        ref_obj = Loss()
        skiplist += dir(ref_obj)
    else:
        raise ValueError(f'Invalid obj_type: {obj_type}')
    global_state.set_global_attribute(f'saving_attr_skiplist_{obj_type}', skiplist)
    return skiplist