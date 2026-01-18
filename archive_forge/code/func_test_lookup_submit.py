from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_submit(self):
    self.assertAliasFromBranch(self.branch.set_submit_branch, 'http://b', ':submit')