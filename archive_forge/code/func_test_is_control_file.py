from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_is_control_file(self):
    mapping = BzrGitMappingv1()
    if mapping.roundtripping:
        self.assertTrue(mapping.is_special_file('.bzrdummy'))
        self.assertTrue(mapping.is_special_file('.bzrfileids'))
    self.assertFalse(mapping.is_special_file('.bzrfoo'))