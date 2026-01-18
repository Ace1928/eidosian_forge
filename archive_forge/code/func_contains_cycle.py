import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def contains_cycle(self):
    """Check whether there are cycles.

        >>> dg = DependencyGraph(treebank_data)
        >>> dg.contains_cycle()
        False

        >>> cyclic_dg = DependencyGraph()
        >>> top = {'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0}
        >>> child1 = {'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1}
        >>> child2 = {'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2}
        >>> child3 = {'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3}
        >>> child4 = {'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4}
        >>> cyclic_dg.nodes = {
        ...     0: top,
        ...     1: child1,
        ...     2: child2,
        ...     3: child3,
        ...     4: child4,
        ... }
        >>> cyclic_dg.root = top

        >>> cyclic_dg.contains_cycle()
        [1, 2, 4, 3]

        """
    distances = {}
    for node in self.nodes.values():
        for dep in node['deps']:
            key = tuple([node['address'], dep])
            distances[key] = 1
    for _ in self.nodes:
        new_entries = {}
        for pair1 in distances:
            for pair2 in distances:
                if pair1[1] == pair2[0]:
                    key = tuple([pair1[0], pair2[1]])
                    new_entries[key] = distances[pair1] + distances[pair2]
        for pair in new_entries:
            distances[pair] = new_entries[pair]
            if pair[0] == pair[1]:
                path = self.get_cycle_path(self.get_by_address(pair[0]), pair[0])
                return path
    return False