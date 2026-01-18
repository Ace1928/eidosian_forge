import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def all_neighborhoods(link, radius):
    nhds = []
    i = 0
    for c in link.crossings:
        print(i)
        i += 1
        link_copy = link.copy()
        c_copy = crossing_by_label(c.label, link_copy)
        T1, T2 = tangle_neighborhood(link_copy, c_copy, radius)
        nhds.append(T1)
    return nhds