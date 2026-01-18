from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_lookup_badvalue(self):
    e = self.assertRaises(UnsetLocationAlias, directories.dereference, ':parent')
    self.assertEqual('No parent location assigned.', str(e))