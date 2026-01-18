import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def all_four_cycles(G):
    """
    Returns all possible four cycles in G
    """
    four_cycles = [x for v in G.vertices for x in all_four_cycles_at_vertex(G, v)]
    four_cycles_no_duplicates = []
    for fc in four_cycles:
        seen_before = False
        for seen_fc in four_cycles_no_duplicates:
            if set(fc) == set(seen_fc):
                seen_before = True
                break
        if not seen_before:
            four_cycles_no_duplicates.append(fc)
    return four_cycles_no_duplicates