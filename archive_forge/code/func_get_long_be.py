import warnings
from collections import namedtuple
def get_long_be(b):
    return b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3]