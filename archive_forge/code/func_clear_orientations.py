from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def clear_orientations(link):
    """
    Resets the orientations on the crossings of a link to default values
    """
    link.link_components = None
    for i in link.crossings:
        i.sign = 0
        i.directions.clear()