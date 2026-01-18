import inspect
import types
from unittest import mock
import pkg_resources
import testtools
import troveclient.shell
def mock_discover_via_contrib_path(self, version):
    yield ('bar', types.ModuleType('bar'))