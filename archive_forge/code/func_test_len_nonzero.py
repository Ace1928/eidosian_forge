import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def test_len_nonzero(self):
    assert not self.q
    self.assertEqual(len(self.q), 0)
    self.q.push(b'a', 3)
    assert self.q
    self.q.push(b'b', 1)
    self.q.push(b'c', 2)
    self.q.push(b'd', 1)
    self.assertEqual(len(self.q), 4)
    self.q.pop()
    self.q.pop()
    self.q.pop()
    self.q.pop()
    assert not self.q
    self.assertEqual(len(self.q), 0)