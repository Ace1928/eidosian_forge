import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def pydot__tree_to_dot(tree: Tree, filename, rankdir='LR', **kwargs):
    graph = pydot__tree_to_graph(tree, rankdir, **kwargs)
    graph.write(filename)