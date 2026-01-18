import re
import warnings
from enum import Enum
from math import gcd
def ewarn(message):
    warnings.warn(message, ExprWarning, stacklevel=2)