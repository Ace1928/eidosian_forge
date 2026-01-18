import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def next_corner(self):
    c, e = (self.crossing, self.strand_index)
    return CrossingStrand(*c.adjacent[(e + 1) % c._adjacent_len])