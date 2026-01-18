import functools
from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils
def parameter_modifiers(self):
    return [test_combinations.OptionalParameter('model_type')]