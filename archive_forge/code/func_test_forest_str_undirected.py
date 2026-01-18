import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_forest_str_undirected():
    graph = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)
    nx.forest_str(graph)
    node_target0 = dedent('\n        ╙── 0\n            ├── 1\n            │   ├── 3\n            │   └── 4\n            └── 2\n                ├── 5\n                └── 6\n        ').strip()
    ret = nx.forest_str(graph, sources=[0])
    print(ret)
    assert ret == node_target0
    node_target2 = dedent('\n        ╙── 2\n            ├── 0\n            │   └── 1\n            │       ├── 3\n            │       └── 4\n            ├── 5\n            └── 6\n        ').strip()
    ret = nx.forest_str(graph, sources=[2])
    print(ret)
    assert ret == node_target2