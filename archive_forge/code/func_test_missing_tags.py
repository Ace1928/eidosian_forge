from breezy import osutils, tests
def test_missing_tags(self):
    """Test showing tags"""
    a_tree = self.make_branch_and_tree('a')
    self.build_tree_contents([('a/a', b'initial\n')])
    a_tree.add('a')
    a_tree.commit(message='initial')
    b_tree = a_tree.controldir.sprout('b').open_workingtree()
    self.build_tree_contents([('b/a', b'initial\nmore\n')])
    b_tree.commit(message='more')
    b_tree.branch.tags.set_tag('a-tag', b_tree.last_revision())
    for log_format in ['long', 'short', 'line']:
        out, err = self.run_bzr(f'missing --log-format={log_format} ../a', working_dir='b', retcode=1)
        self.assertContainsString(out, 'a-tag')
        out, err = self.run_bzr(f'missing --log-format={log_format} ../b', working_dir='a', retcode=1)
        self.assertContainsString(out, 'a-tag')