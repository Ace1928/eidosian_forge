import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def format_adjacent(a):
    return (a[0].label, a[1]) if a else None