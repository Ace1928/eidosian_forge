import importlib
import inspect
from unittest import mock
import stevedore
from stevedore import extension
from novaclient import client
from novaclient.tests.unit import utils
def mock_discover_via_entry_points():
    module_spec = importlib.machinery.ModuleSpec('baz', None)
    module = importlib.util.module_from_spec(module_spec)
    yield ('baz', module)