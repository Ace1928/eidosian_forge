import math
import numbers
import random
import sys
from fractions import Fraction
from decimal import Decimal
from itertools import groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum
from functools import reduce
from operator import mul
from collections import Counter, namedtuple, defaultdict
@classmethod
def from_samples(cls, data):
    """Make a normal distribution instance from sample data."""
    return cls(*_mean_stdev(data))