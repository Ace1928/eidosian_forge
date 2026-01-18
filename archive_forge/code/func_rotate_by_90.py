import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def rotate_by_90(self):
    """Effectively switches the crossing"""
    self.rotate(1)