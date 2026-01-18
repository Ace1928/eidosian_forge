import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def peer_info(self):
    labels = self.strand_labels
    SW = labels[0] if self.sign == 1 else labels[1]
    NW = labels[3] if self.sign == 1 else labels[0]
    if SW % 2 == 0:
        ans = (SW, (-NW, -self.sign))
    else:
        ans = (NW, (SW, self.sign))
    return ans