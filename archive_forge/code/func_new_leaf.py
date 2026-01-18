import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def new_leaf(leaf):
    node = pydot.Node(i[0], label=repr(leaf))
    i[0] += 1
    graph.add_node(node)
    return node