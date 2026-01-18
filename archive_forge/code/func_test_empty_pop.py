import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_empty_pop(self):
    q = MappedQueue()
    pytest.raises(IndexError, q.pop)