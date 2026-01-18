from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_this(self):
    self.assertEqual(self.branch.base, directories.dereference(':this'))