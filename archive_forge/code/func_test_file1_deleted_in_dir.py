import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_file1_deleted_in_dir(self):
    outer, inner, revs = self.make_outer_tree()
    outer.remove(['dir-outer/dir/file1'], keep_files=False)
    outer.commit('delete file1')
    outer.merge_from_branch(inner)
    outer.commit('merge the rest')
    if outer.supports_rename_tracking():
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file2', 'dir-outer/file3', 'foo'], outer)
    else:
        self.assertTreeLayout(['dir', 'dir-outer', 'dir-outer/dir', 'dir-outer/file3', 'dir/file2', 'foo'], outer)