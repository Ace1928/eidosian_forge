import collections
from absl import logging
def set_fn(option, value):
    if not isinstance(value, ty):
        raise TypeError('Property "{}" must be of type {}, got: {} (type: {})'.format(name, ty, value, type(value)))
    option._options[name] = value