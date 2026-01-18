import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def label_crossing(self, comp, labels):
    c, e = (self.crossing, self.strand_index)
    f = (e + 2) % 4
    c.strand_labels[e], c.strand_components[e] = (labels[self], comp)
    c.strand_labels[f], c.strand_components[f] = (labels[self.next()], comp)