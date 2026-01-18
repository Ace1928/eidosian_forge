import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_destroy_group_invalid_state(self):
    """
        Test 'ex_destroy_group' invalid state.

        Test the Driver raises an exception when the group is in
        invalid temporal state.
        """
    self.driver = AbiquoNodeDriver('ve', 'geta', 'http://dummy.host.com/api')
    location = self.driver.list_locations()[0]
    group = self.driver.ex_list_groups(location)[1]
    self.assertRaises(LibcloudError, group.destroy)