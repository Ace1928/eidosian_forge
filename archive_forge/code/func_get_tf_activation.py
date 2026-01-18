import math
import tensorflow as tf
from packaging.version import parse
def get_tf_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f'function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}')