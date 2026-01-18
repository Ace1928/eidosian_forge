import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def dual_graph(self):
    """
        The dual graph to a link diagram D, whose vertices correspond to
        complementary regions (faces) of D and whose edges are dual to the
        edges of D.
        """
    from . import simplify
    return simplify.DualGraphOfFaces(self)