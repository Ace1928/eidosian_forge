from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def insert_strand(X, x):
    Y, y = X.adjacent[x]
    S = Strand()
    S[0], S[1] = (X[x], Y[y])
    return S