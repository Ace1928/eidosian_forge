import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_create_node_no_image_raise_exception(self):
    """
        Test 'create_node' without image.

        Test the 'create_node' function without 'image' parameter raises
        an Exception
        """
    self.assertRaises(TypeError, self.driver.create_node)