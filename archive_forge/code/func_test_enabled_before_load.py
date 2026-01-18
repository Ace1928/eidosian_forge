from unittest import mock
from stevedore import named
from stevedore.tests import utils
def test_enabled_before_load(self):
    init_name = 'stevedore.tests.test_extension.FauxExtension.__init__'
    with mock.patch(init_name) as m:
        m.side_effect = AssertionError
        em = named.NamedExtensionManager('stevedore.test.extension', names=['no-such-extension'], invoke_on_load=True, invoke_args=('a',), invoke_kwds={'b': 'B'})
        actual = em.names()
        self.assertEqual(actual, [])