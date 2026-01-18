from breezy import merge_directive, tests
def test_bundle_info(self):
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/foo'])
    source.add('foo')
    source.commit('added file', rev_id=b'rev1')
    with open('bundle', 'wb') as bundle:
        source.branch.repository.create_bundle(b'rev1', b'null:', bundle, '4')
    info = self.run_bzr('bundle-info bundle')[0]
    self.assertContainsRe(info, 'file: [12] .0 multiparent.')
    self.assertContainsRe(info, 'nicks: source')
    self.assertNotContainsRe(info, 'foo')
    self.run_bzr_error(['--verbose requires a merge directive'], 'bundle-info -v bundle')
    target = self.make_branch('target')
    md = merge_directive.MergeDirective2.from_objects(source.branch.repository, b'rev1', 0, 0, 'target', base_revision_id=b'null:')
    with open('directive', 'wb') as directive:
        directive.writelines(md.to_lines())
    info = self.run_bzr('bundle-info -v directive')[0]
    self.assertContainsRe(info, 'foo')