import itertools
import pytest
import networkx as nx
def dict_to_sets(colors):
    if len(colors) == 0:
        return []
    k = max(colors.values()) + 1
    sets = [set() for _ in range(k)]
    for node, color in colors.items():
        sets[color].add(node)
    return sets