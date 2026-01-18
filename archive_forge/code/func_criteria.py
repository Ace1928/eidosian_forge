from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def criteria(cep):
    return getattr(cep, 'is_' + kind + '_crossing')()