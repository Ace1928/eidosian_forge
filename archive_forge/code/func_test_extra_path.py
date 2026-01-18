from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_extra_path(self):
    self.assertEqual(urlutils.join(self.branch.base, 'arg'), directories.dereference(':this/arg'))