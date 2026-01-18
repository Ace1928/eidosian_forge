from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def common_element(X, Y):
    return next(iter(set(X) & set(Y)))