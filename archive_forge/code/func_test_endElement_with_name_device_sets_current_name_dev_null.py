from tests.compat import unittest
from boto.ec2.connection import EC2Connection
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from tests.compat import OrderedDict
from tests.unit import AWSMockServiceTestCase
def test_endElement_with_name_device_sets_current_name_dev_null(self):
    self.block_device_mapping.endElement('device', '/dev/null', None)
    self.assertEqual(self.block_device_mapping.current_name, '/dev/null')