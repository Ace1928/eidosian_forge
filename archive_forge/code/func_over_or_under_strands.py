from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def over_or_under_strands(link, kind):
    """
    Returns a list of the sequences of (over/under) crossings (which
    are lists of CrossingEntryPoints), sorted in descending order
    of length.
    """

    def criteria(cep):
        return getattr(cep, 'is_' + kind + '_crossing')()
    ceps = OrderedSet([cep for cep in link.crossing_entries() if criteria(cep)])
    strands = []
    while ceps:
        cep = ceps.pop()
        start_crossing = cep.crossing
        is_loop = False
        forward_strand = [cep]
        forward_cep = cep.next()
        while criteria(forward_cep):
            if forward_cep.crossing == start_crossing:
                is_loop = True
                break
            forward_strand.append(forward_cep)
            forward_cep = forward_cep.next()
        backwards_strand = []
        backwards_cep = cep.previous()
        if is_loop:
            strand = forward_strand
        else:
            while criteria(backwards_cep):
                backwards_strand.append(backwards_cep)
                backwards_cep = backwards_cep.previous()
            strand = backwards_strand[::-1]
            strand.extend(forward_strand)
        strands.append(strand)
        ceps.difference_update(strand)
    return sorted(strands, key=len, reverse=True)