import binascii
import functools
import logging
import re
import string
import struct
import numpy as np
from matplotlib.cbook import _format_approx
from . import _api
def is_slash_name(self):
    return self.raw.startswith('/')