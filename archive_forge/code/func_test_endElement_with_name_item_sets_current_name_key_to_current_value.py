from tests.compat import unittest
from boto.ec2.connection import EC2Connection
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from tests.compat import OrderedDict
from tests.unit import AWSMockServiceTestCase
def test_endElement_with_name_item_sets_current_name_key_to_current_value(self):
    self.block_device_mapping.current_name = 'some name'
    self.block_device_mapping.current_value = 'some value'
    self.block_device_mapping.endElement('item', 'some item', None)
    self.assertEqual(self.block_device_mapping['some name'], 'some value')