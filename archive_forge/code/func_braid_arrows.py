from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def braid_arrows(link):
    """
    Helper function to determine positions of all the crossings in a braid
    description of the link.
    """
    link_copy = link.copy()
    isotope_to_braid(link_copy)
    circles = seifert_circles(link_copy)
    tree = seifert_tree(link_copy)
    tails = [e[0] for e in tree]
    heads = [e[1] for e in tree]
    for t in tails:
        if t not in heads:
            start = tails.index(t)
            break
    ordered_strands = [circles[start]]
    for i in range(len(circles) - 1):
        new_tail = tree[start][1]
        start = tails.index(new_tail)
        ordered_strands.append(circles[start])
    positions_in_next_strand = []
    for i in range(len(ordered_strands) - 1):
        for n, cep in enumerate(ordered_strands[i]):
            found_next = False
            for m, next_cep in enumerate(ordered_strands[i + 1]):
                if cep.crossing == next_cep.crossing:
                    ordered_strands[i + 1] = cyclic_permute(ordered_strands[i + 1], m)
                    found_next = True
                    break
            if found_next:
                break
    for i in range(len(ordered_strands) - 1):
        positions = {}
        for n, cep in enumerate(ordered_strands[i]):
            for m, next_cep in enumerate(ordered_strands[i + 1]):
                if cep.crossing == next_cep.crossing:
                    positions[n] = (m, cep.strand_index % 2)
                    break
        positions_in_next_strand.append(positions)
    ordered_strands = ordered_strands[::-1]
    arrows = [[i, positions[i][0], n, positions[i][1]] for n, positions in enumerate(positions_in_next_strand) for i in positions]
    straighten_arrows(arrows)
    arrows = sorted(arrows, key=lambda x: x[0])
    for arrow in arrows:
        arrow.pop(1)
    return arrows