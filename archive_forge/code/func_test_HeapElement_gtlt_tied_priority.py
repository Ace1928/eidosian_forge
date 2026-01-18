import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_HeapElement_gtlt_tied_priority():
    bar = _HeapElement(1, 'a')
    foo = _HeapElement(1, 'b')
    assert foo > bar
    assert bar < foo