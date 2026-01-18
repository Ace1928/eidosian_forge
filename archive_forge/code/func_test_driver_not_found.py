from osprofiler.drivers import base
from osprofiler.tests import test
def test_driver_not_found(self):
    self.assertRaises(ValueError, base.get_driver, 'Driver not found for connection string: nonexisting://')