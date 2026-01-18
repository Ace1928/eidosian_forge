from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def make_simple_branch_with_ghost(self):
    if not self.repository_format.supports_ghosts:
        raise TestNotApplicable('repository format does not support ghosts')
    builder = self.make_branch_builder('source')
    builder.start_series()
    a_revid = builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))])
    b_revid = builder.build_snapshot([a_revid, b'ghost-id'], [])
    builder.finish_series()
    source_b = builder.get_branch()
    source_b.lock_read()
    self.addCleanup(source_b.unlock)
    return (source_b, b_revid)