from octavia_lib.api.drivers import exceptions
from octavia_lib.api.drivers import provider_base as driver_base
from octavia_lib.tests.unit import base
def test_member_batch_update(self):
    self.assertRaises(exceptions.NotImplementedError, self.driver.member_batch_update, False, False)