from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_get_directory_service(self):
    directory, suffix = self.registry.get_prefix('foo:bar')
    self.assertIs(FooService, directory)
    self.assertEqual('bar', suffix)