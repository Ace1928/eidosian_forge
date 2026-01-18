from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_generate_file_id(self):
    mapping = BzrGitMappingv1()
    self.assertIsInstance(mapping.generate_file_id('la'), bytes)
    self.assertIsInstance(mapping.generate_file_id('é'), bytes)