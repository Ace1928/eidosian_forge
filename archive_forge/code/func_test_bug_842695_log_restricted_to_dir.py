import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_bug_842695_log_restricted_to_dir(self):
    trunk = self.make_branch_and_tree('this')
    trunk.commit('initial trunk')
    adder = trunk.controldir.sprout('adder').open_workingtree()
    merger = trunk.controldir.sprout('merger').open_workingtree()
    self.build_tree_contents([('adder/dir/',), ('adder/dir/file', b'foo')])
    adder.add(['dir', 'dir/file'])
    adder.commit('added dir')
    trunk.merge_from_branch(adder.branch)
    trunk.commit('merged adder into trunk')
    merger.merge_from_branch(trunk.branch)
    merger.commit('merged trunk into merger')
    for i in range(200):
        trunk.commit('intermediate commit %d' % i)
    trunk.merge_from_branch(merger.branch)
    trunk.commit('merged merger into trunk')
    file_id = trunk.path2id('dir')
    lf = LogCatcher()
    lf.supports_merge_revisions = True
    log.show_log(trunk.branch, lf, file_id)
    try:
        self.assertEqual(['2', '1.1.1'], [r.revno for r in lf.revisions])
    except AssertionError:
        raise tests.KnownFailure('bug #842695')