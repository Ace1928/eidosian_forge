import re
import warnings
import keras_tuner
import tensorflow as tf
from packaging.version import parse
from tensorflow import nest
def check_kt_version() -> None:
    if parse(keras_tuner.__version__) < parse('1.1.0'):
        warnings.warn(f'The Keras Tuner package version needs to be at least 1.1.0 \nfor AutoKeras to run. Currently, your Keras Tuner version is \n{keras_tuner.__version__}. Please upgrade with \n`$ pip install --upgrade keras-tuner`. \nYou can use `pip freeze` to check afterwards that everything is ok.', ImportWarning)