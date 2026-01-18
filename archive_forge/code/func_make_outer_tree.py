import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def make_outer_tree(self):
    outer = self.make_branch_and_tree('outer')
    self.build_tree_contents([('outer/foo', b'foo')])
    outer.add('foo')
    outer.commit('added foo')
    inner, revs = self.make_inner_branch()
    outer.merge_from_branch(inner, to_revision=revs[0], from_revision=b'null:')
    if outer.supports_setting_file_ids():
        outer.set_root_id(outer.basis_tree().path2id(''))
    outer.commit('merge inner branch')
    outer.mkdir('dir-outer')
    outer.move(['dir', 'file3'], to_dir='dir-outer')
    outer.commit('rename imported dir and file3 to dir-outer')
    return (outer, inner, revs)