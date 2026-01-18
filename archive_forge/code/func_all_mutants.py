import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def all_mutants(link):
    G = link.dual_graph()
    return [mutate(link, four_cycle) for four_cycle in all_four_cycles(G)]