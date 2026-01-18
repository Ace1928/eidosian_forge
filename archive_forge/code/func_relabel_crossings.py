from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def relabel_crossings(link):
    """
    Relabel the crossings as integers
    """
    for i, cr in enumerate(link.crossings):
        cr.label = str(i)