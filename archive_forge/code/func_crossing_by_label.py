import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def crossing_by_label(label, link):
    for c in link.crossings:
        if c.label == label:
            return c