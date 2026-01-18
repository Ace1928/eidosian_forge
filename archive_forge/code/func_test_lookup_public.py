from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_public(self):
    self.assertAliasFromBranch(self.branch.set_public_branch, 'http://c', ':public')