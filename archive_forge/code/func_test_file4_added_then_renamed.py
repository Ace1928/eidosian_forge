import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_file4_added_then_renamed(self):
    outer, inner, revs = self.make_outer_tree()
    nb_conflicts = outer.merge_from_branch(inner, to_revision=revs[3])
    if outer.supports_rename_tracking():
        self.assertEqual(1, len(nb_conflicts))
    else:
        self.assertEqual(0, len(nb_conflicts))
    try:
        outer.set_conflicts([])
    except errors.UnsupportedOperation:
        pass
    outer.commit('added file4')
    nb_conflicts = outer.merge_from_branch(inner, to_revision=revs[4])
    if outer.supports_rename_tracking():
        self.assertEqual(1, len(nb_conflicts))
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/dir/file4', 'dir-outer/file3', 'foo'], outer)
    else:
        if outer.has_versioned_directories():
            self.assertEqual(2, len(nb_conflicts))
        else:
            self.assertEqual(0, len(nb_conflicts))
        self.assertTreeLayout(['dir', 'dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/file3', 'dir/file4', 'foo'], outer)