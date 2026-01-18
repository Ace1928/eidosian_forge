import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def sum_logs(logs):
    return reduce(add_logs, logs[1:], logs[0]) if len(logs) != 0 else _NINF