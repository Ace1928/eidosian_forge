from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def reverse_type_II(link, c, d, label1, label2, rebuild=False):
    """
    Cross two strands defined by two crossing strands c and d in the same face.
    """
    new1, new2 = (Crossing(label1), Crossing(label2))
    c_cross, c_ep = (c.crossing, c.strand_index)
    cop_cross, cop_ep = (c.opposite().crossing, c.opposite().strand_index)
    d_cross, d_ep = (d.crossing, d.strand_index)
    dop_cross, dop_ep = (d.opposite().crossing, d.opposite().strand_index)
    new1[2], new1[3] = (new2[0], new2[3])
    new1[0], new1[1] = (dop_cross[dop_ep], c_cross[c_ep])
    new2[1], new2[2] = (cop_cross[cop_ep], d_cross[d_ep])
    link.crossings.append(new1)
    link.crossings.append(new2)
    if rebuild:
        link._rebuild(same_components_and_orientations=True)