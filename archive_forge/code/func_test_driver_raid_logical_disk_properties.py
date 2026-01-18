from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
def test_driver_raid_logical_disk_properties(self):
    properties = self.mgr.raid_logical_disk_properties(DRIVER2['name'])
    expect = [('GET', '/v1/drivers/%s/raid/logical_disk_properties' % DRIVER2['name'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(DRIVER2_RAID_LOGICAL_DISK_PROPERTIES, properties)