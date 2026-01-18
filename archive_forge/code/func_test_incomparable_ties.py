import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_incomparable_ties(self):
    d = {5: 0, 4: 0, 'a': 0, 2: 0, 1: 0}
    pytest.raises(TypeError, MappedQueue, d)