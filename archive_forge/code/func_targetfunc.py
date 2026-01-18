import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def targetfunc(x):
    return gamma * len(population) * math.log(2) / halflifefunc(x) * math.exp(-math.log(2) * (x - cutoffsize) / halflifefunc(x))