import re
import operator
from fractions import Fraction
import sys
def safe_dict(d, var):
    if var in d:
        return d[var]
    else:
        return 0