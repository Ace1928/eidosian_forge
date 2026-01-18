from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def seifert_circles(link):
    """
    Returns the circles in the diagram created by Seifert's algorithm

    >>> fig8 = [(1,7,2,6),(5,3,6,2),(7,4,0,5),(3,0,4,1)]
    >>> L = Link(fig8)
    >>> sorted(len(C) for C in seifert_circles(L))
    [2, 2, 4]
    """
    ceps = OrderedSet(link.crossing_entries())
    circles = []
    while ceps:
        start_cep = ceps.pop()
        circle = [start_cep]
        cep = start_cep.other().next()
        while cep != start_cep:
            circle.append(cep)
            ceps.remove(cep)
            cep = cep.other().next()
        circles.append(circle)
    return circles