from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_genlet_simple(self):
    for g in (g1, g2, g3):
        seen = []
        for _ in range(3):
            for j in g(5, seen):
                seen.append(j)
        self.assertEqual(seen, 3 * [1, 0, 2, 1, 3, 2, 4, 3, 5, 4])