import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_destroy_node_response_failed(self):
    """
        'destroy_node' asynchronous error.

        Test that the driver handles correctly when, for some reason,
        the 'destroy' job fails.
        """
    self.driver = AbiquoNodeDriver('muten', 'roshi', 'http://dummy.host.com/api')
    node = self.driver.list_nodes()[0]
    ret = self.driver.destroy_node(node)
    self.assertFalse(ret)