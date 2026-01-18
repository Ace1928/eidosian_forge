import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def random_tree_knot(size, simplify=None, prime_decomp=False):
    found_nontrivial = False
    while not found_nontrivial:
        link = random_tree_link(size)
        knot = link.sublink(max(link.link_components, key=len))
        knot.simplify(mode=simplify)
        if len(knot) > 0:
            found_nontrivial = True
    if prime_decomp:
        cant_deconnect = False
        while cant_deconnect:
            ds = knot.deconnect_sum()
            knot = max(ds, key=len)
            cant_deconnect = len(ds) > 1
    return knot