import platform
import warnings
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src.optimizers import adadelta
from keras.src.optimizers import adafactor
from keras.src.optimizers import adagrad
from keras.src.optimizers import adam
from keras.src.optimizers import adamax
from keras.src.optimizers import adamw
from keras.src.optimizers import ftrl
from keras.src.optimizers import lion
from keras.src.optimizers import nadam
from keras.src.optimizers import optimizer as base_optimizer
from keras.src.optimizers import rmsprop
from keras.src.optimizers import sgd
from keras.src.optimizers.legacy import adadelta as adadelta_legacy
from keras.src.optimizers.legacy import adagrad as adagrad_legacy
from keras.src.optimizers.legacy import adam as adam_legacy
from keras.src.optimizers.legacy import adamax as adamax_legacy
from keras.src.optimizers.legacy import ftrl as ftrl_legacy
from keras.src.optimizers.legacy import gradient_descent as gradient_descent_legacy
from keras.src.optimizers.legacy import nadam as nadam_legacy
from keras.src.optimizers.legacy import optimizer_v2 as base_optimizer_legacy
from keras.src.optimizers.legacy import rmsprop as rmsprop_legacy
from keras.src.optimizers.legacy.adadelta import Adadelta
from keras.src.optimizers.legacy.adagrad import Adagrad
from keras.src.optimizers.legacy.adam import Adam
from keras.src.optimizers.legacy.adamax import Adamax
from keras.src.optimizers.legacy.ftrl import Ftrl
from keras.src.optimizers.legacy.gradient_descent import SGD
from keras.src.optimizers.legacy.nadam import Nadam
from keras.src.optimizers.legacy.rmsprop import RMSprop
from keras.src.optimizers.optimizer_v1 import Optimizer
from keras.src.optimizers.optimizer_v1 import TFOptimizer
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from tensorflow.python.util.tf_export import keras_export
def is_arm_mac():
    return platform.system() == 'Darwin' and platform.processor() == 'arm'