from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_default(self):
    default = self.make_branch('.')
    non_default = default.controldir.create_branch(name='nondefault')
    self.assertEqual(urlutils.join_segment_parameters(default.controldir.user_url, {'branch': ''}), directories.dereference('co:'))