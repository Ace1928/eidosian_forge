import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_HeapElement_iter():
    foo = _HeapElement(1, 'a')
    bar = _HeapElement(1.1, (3, 2, 1))
    assert list(foo) == [1, 'a']
    assert list(bar) == [1.1, 3, 2, 1]