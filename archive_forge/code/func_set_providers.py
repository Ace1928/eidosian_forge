from pytest import approx
from networkx import is_connected, neighbors
from networkx.generators.internet_as_graphs import random_internet_as_graph
@classmethod
def set_providers(cls, i):
    if i not in cls.providers:
        cls.providers[i] = set()
        for j in neighbors(cls.G, i):
            e = cls.G.edges[i, j]
            if e['type'] == 'transit':
                customer = int(e['customer'])
                if i == customer:
                    cls.set_providers(j)
                    cls.providers[i] = cls.providers[i].union(cls.providers[j])
                    cls.providers[i].add(j)
                elif j != customer:
                    raise ValueError('Inconsistent data in the graph edge attributes')