import warnings
from collections import namedtuple
def get_short_le(b):
    return b[1] << 8 | b[0]