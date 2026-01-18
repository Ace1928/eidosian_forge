import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_create_group_location_does_not_exist(self):
    """
        Test 'create_node' with an unexistent location.

        Defines a 'fake' location and tries to create a node into it.
        """
    location = NodeLocation(435, 'fake-location', 'Spain', self.driver)
    self.assertRaises(LibcloudError, self.driver.ex_create_group, name='new_group_name', location=location)