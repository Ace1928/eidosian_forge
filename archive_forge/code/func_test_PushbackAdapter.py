import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_PushbackAdapter():
    it = PushbackAdapter(iter([1, 2, 3, 4]))
    assert it.has_more()
    assert six.advance_iterator(it) == 1
    it.push_back(0)
    assert six.advance_iterator(it) == 0
    assert six.advance_iterator(it) == 2
    assert it.peek() == 3
    it.push_back(10)
    assert it.peek() == 10
    it.push_back(20)
    assert it.peek() == 20
    assert it.has_more()
    assert list(it) == [20, 10, 3, 4]
    assert not it.has_more()