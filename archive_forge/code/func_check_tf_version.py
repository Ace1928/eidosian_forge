import re
import warnings
import keras_tuner
import tensorflow as tf
from packaging.version import parse
from tensorflow import nest
def check_tf_version() -> None:
    if parse(tf.__version__) < parse('2.7.0'):
        warnings.warn(f'The Tensorflow package version needs to be at least 2.7.0 \nfor AutoKeras to run. Currently, your TensorFlow version is \n{tf.__version__}. Please upgrade with \n`$ pip install --upgrade tensorflow`. \nYou can use `pip freeze` to check afterwards that everything is ok.', ImportWarning)