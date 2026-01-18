from octavia_lib.api.drivers import exceptions
from octavia_lib.api.drivers import provider_base as driver_base
from octavia_lib.tests.unit import base
def test_l7rule_delete(self):
    self.assertRaises(exceptions.NotImplementedError, self.driver.l7rule_delete, False)