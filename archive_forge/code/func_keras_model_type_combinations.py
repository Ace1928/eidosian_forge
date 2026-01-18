import functools
from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils
def keras_model_type_combinations():
    return combinations.combine(model_type=KERAS_MODEL_TYPES)