import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_forest_str_directed():
    graph = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    for node in graph.nodes:
        graph.nodes[node]['label'] = 'node_' + chr(ord('a') + node)
    node_target = dedent('\n        ╙── 0\n            ├─╼ 1\n            │   ├─╼ 3\n            │   └─╼ 4\n            └─╼ 2\n                ├─╼ 5\n                └─╼ 6\n        ').strip()
    label_target = dedent('\n        ╙── node_a\n            ├─╼ node_b\n            │   ├─╼ node_d\n            │   └─╼ node_e\n            └─╼ node_c\n                ├─╼ node_f\n                └─╼ node_g\n        ').strip()
    ret = nx.forest_str(graph, with_labels=False)
    print(ret)
    assert ret == node_target
    ret = nx.forest_str(graph, with_labels=True)
    print(ret)
    assert ret == label_target
    lines = []
    ret = nx.forest_str(graph, write=lines.append, with_labels=False)
    assert ret is None
    assert lines == node_target.split('\n')
    ret = nx.forest_str(graph, write=print)
    assert ret is None