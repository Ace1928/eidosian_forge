from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_escape_underscore_space(self):
    self.assertEqual(b'bla___s', escape_file_id(b'bla_ '))