import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def test_close_return_active(self):
    self.q.push(b'b', 1)
    self.q.push(b'c', 2)
    self.q.push(b'a', 3)
    self.q.pop()
    self.assertEqual(sorted(self.q.close()), [2, 3])