import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def all_crossings_oriented(self):
    return len([c for c in self.crossings if c.sign == 0]) == 0