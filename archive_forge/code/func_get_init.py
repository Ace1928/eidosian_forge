import re
import sys
import inspect
import operator
import itertools
import collections
from inspect import getfullargspec
def get_init(cls):
    return cls.__init__