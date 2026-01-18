import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_forest_str_undirected_multi_tree():
    tree1 = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)
    tree2 = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)
    tree2 = nx.relabel_nodes(tree2, {n: n + len(tree1) for n in tree2.nodes})
    forest = nx.union(tree1, tree2)
    ret = nx.forest_str(forest, sources=[0, 7])
    print(ret)
    target = dedent('\n        ╟── 0\n        ╎   ├── 1\n        ╎   │   ├── 3\n        ╎   │   └── 4\n        ╎   └── 2\n        ╎       ├── 5\n        ╎       └── 6\n        ╙── 7\n            ├── 8\n            │   ├── 10\n            │   └── 11\n            └── 9\n                ├── 12\n                └── 13\n        ').strip()
    assert ret == target
    ret = nx.forest_str(forest, sources=[0, 7], ascii_only=True)
    print(ret)
    target = dedent('\n        +-- 0\n        :   |-- 1\n        :   |   |-- 3\n        :   |   L-- 4\n        :   L-- 2\n        :       |-- 5\n        :       L-- 6\n        +-- 7\n            |-- 8\n            |   |-- 10\n            |   L-- 11\n            L-- 9\n                |-- 12\n                L-- 13\n        ').strip()
    assert ret == target