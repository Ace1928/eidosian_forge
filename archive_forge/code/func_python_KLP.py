import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def python_KLP(L):
    vertices = list(L.crossings)
    if len(vertices) == 0:
        return [0, 1, 1, []]
    for i, v in enumerate(vertices):
        v._KLP_index = i
    return [len(vertices), 0, len(L.link_components), [KLPCrossing(c) for c in vertices]]