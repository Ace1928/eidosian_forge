import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def arg_repr(arg_name):
    return '='.join([arg_name, repr(getattr(self, '_%s' % arg_name))])