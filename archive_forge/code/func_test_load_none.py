from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit import utils as test_utils
def test_load_none(self):
    self.assertRaises(exceptions.MissingRequiredOptions, self.create)