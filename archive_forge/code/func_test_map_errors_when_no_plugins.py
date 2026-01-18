import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_map_errors_when_no_plugins(self):
    expected_str = 'No stevedore.test.extension.none extensions found'

    def mapped(ext, *args, **kwds):
        pass
    em = extension.ExtensionManager('stevedore.test.extension.none', invoke_on_load=True)
    try:
        em.map(mapped, 1, 2, a='A', b='B')
    except exception.NoMatches as err:
        self.assertEqual(expected_str, str(err))