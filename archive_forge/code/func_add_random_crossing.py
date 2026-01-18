import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def add_random_crossing(self, label):
    """
    Randomly chooses position on boundary of the tangle and splits into a new
    crossing.
    """
    tangle_copy = self.copy()
    adj = tangle_copy.adjacent
    adj[len(adj) / 2:] = reversed(adj[len(adj) / 2:])
    new_crossing = spherogram.Crossing(label)
    old_position = randint(0, len(adj) - 1)
    old_crossing, old_strand = adj.pop(old_position)
    new_strand = randint(0, 3)
    old_crossing[old_strand] = new_crossing[new_strand]
    for i in range(1, 4):
        adj.insert(old_position, (new_crossing, (new_strand - i) % 4))
    adj[len(adj) / 2:] = reversed(adj[len(adj) / 2:])
    tangle_copy.crossings.append(new_crossing)
    tangle_copy.n = self.n + 1
    return tangle_copy