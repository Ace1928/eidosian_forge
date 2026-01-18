import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def opposite(self):
    """
        The CrossingStrand at the other end of the edge from self
        """
    return CrossingStrand(*self.crossing.adjacent[self.strand_index])