from breezy import conflicts, tests, workingtree
from breezy.tests import features, script
def make_tree_with_conflicts(test, this_path='this', other_path='other', prefix='my'):
    this_tree = test.make_branch_and_tree(this_path)
    test.build_tree_contents([('{}/{}file'.format(this_path, prefix), b'this content\n'), ('{}/{}_other_file'.format(this_path, prefix), b'this content\n'), ('{}/{}dir/'.format(this_path, prefix),)])
    this_tree.add(prefix + 'file')
    this_tree.add(prefix + '_other_file')
    this_tree.add(prefix + 'dir')
    this_tree.commit(message='new')
    other_tree = this_tree.controldir.sprout(other_path).open_workingtree()
    test.build_tree_contents([('{}/{}file'.format(other_path, prefix), b'contentsb\n'), ('{}/{}_other_file'.format(other_path, prefix), b'contentsb\n')])
    other_tree.rename_one(prefix + 'dir', prefix + 'dir2')
    other_tree.commit(message='change')
    test.build_tree_contents([('{}/{}file'.format(this_path, prefix), b'contentsa2\n'), ('{}/{}_other_file'.format(this_path, prefix), b'contentsa2\n')])
    this_tree.rename_one(prefix + 'dir', prefix + 'dir3')
    this_tree.commit(message='change')
    this_tree.merge_from_branch(other_tree.branch)
    return (this_tree, other_tree)