import re
import sys
import inspect
import operator
import itertools
import collections
from inspect import getfullargspec
def dispatch_info(*types):
    """
            An utility to introspect the dispatch algorithm
            """
    check(types)
    lst = [tuple((a.__name__ for a in anc)) for anc in itertools.product(*ancestors(*types))]
    return lst