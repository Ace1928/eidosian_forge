import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def make_branch_with_many_merges(self, path='.', format=None):
    builder = branchbuilder.BranchBuilder(self.get_transport())
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', ''))], revision_id=b'1')
    builder.build_snapshot([b'1'], [], revision_id=b'2')
    builder.build_snapshot([b'1'], [], revision_id=b'1.1.1')
    builder.build_snapshot([b'2'], [], revision_id=b'2.1.1')
    builder.build_snapshot([b'2', b'1.1.1'], [], revision_id=b'3')
    builder.build_snapshot([b'2.1.1'], [], revision_id=b'2.1.2')
    builder.build_snapshot([b'2.1.1'], [], revision_id=b'2.2.1')
    builder.build_snapshot([b'2.1.2', b'2.2.1'], [], revision_id=b'2.1.3')
    builder.build_snapshot([b'3', b'2.1.3'], [], revision_id=b'4')
    builder.build_snapshot([b'4', b'2.1.2'], [], revision_id=b'5')
    builder.finish_series()
    return builder