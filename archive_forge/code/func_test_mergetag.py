from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_mergetag(self):
    c = Commit()
    c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
    c.message = b'Some message'
    c.committer = b'Committer'
    c.commit_time = 4
    c.author_time = 5
    c.commit_timezone = 60 * 5
    c.author_timezone = 60 * 3
    c.author = b'Author'
    tag = make_object(Tag, tagger=b'Jelmer Vernooij <jelmer@samba.org>', name=b'0.1', message=None, object=(Blob, b'd80c186a03f423a81b39df39dc87fd269736ca86'), tag_time=423423423, tag_timezone=0)
    c.mergetag = [tag]
    mapping = BzrGitMappingv1()
    rev, roundtrip_revid, verifiers = mapping.import_commit(c, mapping.revision_id_foreign_to_bzr)
    self.assertEqual(rev.properties['git-mergetag-0'].encode('utf-8'), tag.as_raw_string())