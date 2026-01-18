from tests.compat import mock, unittest
from boto.ec2.address import Address
def check_that_attribute_has_been_set(self, name, value, attribute):
    self.address.endElement(name, value, None)
    self.assertEqual(getattr(self.address, attribute), value)