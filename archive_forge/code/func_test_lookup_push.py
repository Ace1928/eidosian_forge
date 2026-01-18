from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_push(self):
    self.assertAliasFromBranch(self.branch.set_push_location, 'http://e', ':push')