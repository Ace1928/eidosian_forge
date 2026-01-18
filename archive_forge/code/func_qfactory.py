import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def qfactory(self, prio):
    path = os.path.join(self.qdir, str(prio))
    return track_closed(LifoSQLiteQueue)(path)