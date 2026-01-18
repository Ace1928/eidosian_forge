from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def assertRoundtripRevision(self, orig_rev):
    commit = self.mapping.export_commit(orig_rev, b'mysha', self._lookup_parent, True, b'testamentsha')
    rev, roundtrip_revid, verifiers = self.mapping.import_commit(commit, self.mapping.revision_id_foreign_to_bzr, strict=True)
    self.assertEqual(rev.revision_id, self.mapping.revision_id_foreign_to_bzr(commit.id))
    if self.mapping.roundtripping:
        self.assertEqual({'testament3-sha1': b'testamentsha'}, verifiers)
        self.assertEqual(orig_rev.revision_id, roundtrip_revid)
        self.assertEqual(orig_rev.properties, rev.properties)
        self.assertEqual(orig_rev.committer, rev.committer)
        self.assertEqual(orig_rev.timestamp, rev.timestamp)
        self.assertEqual(orig_rev.timezone, rev.timezone)
        self.assertEqual(orig_rev.message, rev.message)
        self.assertEqual(list(orig_rev.parent_ids), list(rev.parent_ids))
    else:
        self.assertEqual({}, verifiers)
    return commit