import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def assertTreeLayout(self, expected, tree):
    with tree.lock_read():
        actual = [e[0] for e in tree.list_files()]
        actual = sorted(actual)
        self.assertEqual(expected, actual)