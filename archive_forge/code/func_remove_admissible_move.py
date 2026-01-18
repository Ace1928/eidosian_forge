from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def remove_admissible_move(link):
    """
    Performs a Reidemester II move to remove one branching point of the Seifert
    tree.  The goal is to turn the Seifert tree into a chain.
    """
    circles = seifert_circles(link)
    moves, circle_pairs = admissible_moves(link)
    tree = seifert_tree(link)
    found_move = False
    for e1, e2 in combinations(tree, 2):
        if e1[0] == e2[0]:
            circles = set([tree.index(e1), tree.index(e2)])
            found_move = True
        elif e1[1] == e2[1]:
            circles = set([tree.index(e1), tree.index(e2)])
            found_move = True
        if found_move:
            move_possible = False
            for n, pair in enumerate(circle_pairs):
                if set(pair) == circles:
                    cs1, cs2 = moves[n]
                    move_possible = True
                    break
            if move_possible:
                label1 = 'n' + str(cs1.crossing.label)
                label2 = 'n' + str(cs2.crossing.label)
                reverse_type_II(link, cs1, cs2, label1, label2)
                link._rebuild(same_components_and_orientations=True)
                break
            else:
                found_move = False
    return found_move