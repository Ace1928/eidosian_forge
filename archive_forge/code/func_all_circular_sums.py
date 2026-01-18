import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def all_circular_sums(self, other):
    """
    All possible tangle sums as above
    """
    if len(self.adjacent) != len(other.adjacent):
        raise Exception('Tangles do not have the same number of strands')
    return [self.circular_sum(other, n) for n in range(len(other.adjacent))]