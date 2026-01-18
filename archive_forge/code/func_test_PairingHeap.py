import pytest
import networkx as nx
from networkx.utils import BinaryHeap, PairingHeap
def test_PairingHeap():
    _test_heap_class(PairingHeap)