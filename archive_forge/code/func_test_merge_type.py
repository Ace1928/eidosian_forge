import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_merge_type(self):
    this = self.make_branch_and_tree('this')
    self.build_tree_contents([('this/foo', b'foo')])
    this.add('foo')
    this.commit('added foo')
    other = this.controldir.sprout('other').open_workingtree()
    self.build_tree_contents([('other/foo', b'bar')])
    other.commit('content -> bar')
    self.build_tree_contents([('this/foo', b'baz')])
    this.commit('content -> baz')

    class QuxMerge(merge.Merge3Merger):

        def text_merge(self, trans_id, paths):
            self.tt.create_file([b'qux'], trans_id)
    this.merge_from_branch(other.branch, merge_type=QuxMerge)
    self.assertEqual(b'qux', this.get_file_text('foo'))