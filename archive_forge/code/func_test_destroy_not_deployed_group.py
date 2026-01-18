import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_destroy_not_deployed_group(self):
    """
        Test 'ex_destroy_group' when group is not deployed.
        """
    location = self.driver.list_locations()[0]
    group = self.driver.ex_list_groups(location)[1]
    self.assertTrue(group.destroy())