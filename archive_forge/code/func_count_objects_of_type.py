import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def count_objects_of_type(_type):
    return sum((1 for obj in gc.get_objects() if not isinstance(obj, weakref.ProxyTypes) and isinstance(obj, _type)))