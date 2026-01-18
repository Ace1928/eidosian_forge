from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def reverse_type_I(link, crossing_strand, label, hand, rebuild=False):
    """
    Add a loop on the strand at crossing_strand with a given label and
    'handedness' hand (twisting left or right).
    """
    D = Crossing(label)
    link.crossings.append(D)
    cs1 = crossing_strand
    cs2 = cs1.opposite()
    if hand == 'left':
        D[1] = D[2]
        cs1ec, cs1cep = (cs1.crossing, cs1.strand_index)
        D[0] = cs1ec[cs1cep]
        cs2ec, cs2cep = (cs2.crossing, cs2.strand_index)
        D[3] = cs2ec[cs2cep]
    else:
        D[2] = D[3]
        cs1ec, cs1cep = (cs1.crossing, cs1.strand_index)
        D[0] = cs1ec[cs1cep]
        cs2ec, cs2cep = (cs2.crossing, cs2.strand_index)
        D[1] = cs2ec[cs2cep]
    if rebuild:
        link._rebuild(same_components_and_orientations=True)