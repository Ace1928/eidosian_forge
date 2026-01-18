import importlib.metadata as importlib_metadata
from stevedore import driver
from stevedore import exception
from stevedore import extension
from stevedore.tests import test_extension
from stevedore.tests import utils
def test_driver_property_invoked_on_load(self):
    em = driver.DriverManager('stevedore.test.extension', 't1', invoke_on_load=True)
    d = em.driver
    self.assertIsInstance(d, test_extension.FauxExtension)