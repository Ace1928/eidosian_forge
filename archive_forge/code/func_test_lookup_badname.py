from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_badname(self):
    e = self.assertRaises(InvalidLocationAlias, directories.dereference, ':booga')
    self.assertEqual('":booga" is not a valid location alias.', str(e))