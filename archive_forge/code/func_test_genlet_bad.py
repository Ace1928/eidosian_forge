from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_genlet_bad(self):
    try:
        Yield(10)
    except RuntimeError:
        pass