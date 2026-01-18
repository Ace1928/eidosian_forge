import functools
from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils
def context_managers(self, kwargs):
    model_type = kwargs.pop('model_type', None)
    if model_type in KERAS_MODEL_TYPES:
        return [testing_utils.model_type_scope(model_type)]
    else:
        return []