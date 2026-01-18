from unittest import mock
from stevedore import named
from stevedore.tests import utils
def test_load_fail_ignored_when_sorted(self):
    em = named.NamedExtensionManager('stevedore.test.extension', names=['e1', 't2', 'e2', 't1'], name_order=True, invoke_on_load=True, invoke_args=('a',), invoke_kwds={'b': 'B'})
    actual = em.names()
    self.assertEqual(['t2', 't1'], actual)
    em = named.NamedExtensionManager('stevedore.test.extension', names=['e1', 't1'], name_order=False, invoke_on_load=True, invoke_args=('a',), invoke_kwds={'b': 'B'})
    actual = em.names()
    self.assertEqual(['t1'], actual)