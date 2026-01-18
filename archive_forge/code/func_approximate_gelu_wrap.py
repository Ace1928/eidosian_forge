import math
import tensorflow as tf
from packaging.version import parse
def approximate_gelu_wrap(x):
    return keras.activations.gelu(x, approximate=True)