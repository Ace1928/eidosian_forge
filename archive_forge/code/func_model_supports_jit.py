import platform
import warnings
import tree
from keras.src import backend
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import optimizers
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.saving import serialization_lib
from keras.src.trainers.compile_utils import CompileLoss
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
def model_supports_jit(model):
    if platform.system() == 'Darwin' and 'arm' in platform.processor().lower():
        if backend.backend() == 'tensorflow':
            from keras.src.utils.module_utils import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                return False
    if all((x.supports_jit for x in model._flatten_layers())):
        return True
    return False