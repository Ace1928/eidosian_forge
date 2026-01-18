from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def random_reverse_type_I(link, label, rebuild=False):
    """
    Randomly adds a loop in a strand, adding one crossing with given label
    """
    cs = random.choice(link.crossing_strands())
    lr = random.choice(['left', 'right'])
    reverse_type_I(link, cs, label, lr, rebuild=rebuild)