import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_create_node_specify_wrong_location(self):
    """
        Test you can not create a node with wrong location.
        """
    image = self.driver.list_images()[0]
    location = NodeLocation(435, 'fake-location', 'Spain', self.driver)
    self.assertRaises(LibcloudError, self.driver.create_node, image=image, location=location)