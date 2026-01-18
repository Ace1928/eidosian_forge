from unittest import mock
from testtools.matchers import GreaterThan
from stevedore import extension
from stevedore import named
from stevedore.tests import utils
def test_extension_failure_custom_callback(self):
    errors = []

    def failure_callback(manager, entrypoint, error):
        errors.append((manager, entrypoint, error))
    em = extension.ExtensionManager('stevedore.test.extension', invoke_on_load=True, on_load_failure_callback=failure_callback)
    extensions = list(em.extensions)
    self.assertTrue(len(extensions), GreaterThan(0))
    self.assertEqual(len(errors), 2)
    for manager, entrypoint, error in errors:
        self.assertIs(manager, em)
        self.assertIsInstance(error, (IOError, ImportError))