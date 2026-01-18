from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def remove_strand(link, strand):
    """
    Delete an overstrand or understrand from a link.  If the strand is
    a loop, it doesn't leave any loose strands and removes the
    loop. Otherwise, there will be two strands left in the link not
    attached to anything.  This function assumes that the start and
    end of the strand are not places where strands crosses itself.
    """
    crossings_seen = [s.crossing for s in strand]
    crossing_set = set()
    for c in crossings_seen:
        if c in crossing_set:
            crossing_set.remove(c)
        else:
            crossing_set.add(c)
    start_cep = strand[0].previous()
    end_cep = strand[-1].next()
    bridge_strands = {c: Strand('strand' + str(c.label)) for c in crossing_set}
    for cep in strand:
        c = cep.crossing
        if c not in crossing_set:
            continue
        right_cs = cep.rotate(1).opposite()
        left_cs = cep.rotate(3).opposite()
        if right_cs.crossing in crossing_set:
            signs_equal = c.sign == right_cs.crossing.sign
            bridge_strands[c][0] = bridge_strands[right_cs.crossing][signs_equal]
        else:
            bridge_strands[c][0] = right_cs.crossing[right_cs.strand_index]
        if left_cs.crossing in crossing_set:
            signs_equal = c.sign == left_cs.crossing.sign
            bridge_strands[c][1] = bridge_strands[left_cs.crossing][1 - signs_equal]
        else:
            bridge_strands[c][1] = left_cs.crossing[left_cs.strand_index]
    remove_crossings(link, set(crossings_seen))
    for s in bridge_strands.values():
        s.fuse()
    return set(crossings_seen)