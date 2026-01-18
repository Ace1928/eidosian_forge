import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_forbidden_controlled(self):
    """
        Test the Forbidden Exception is Controlled.

        Test, through the 'list_images' method, that a '403 Forbidden'
        raises an 'ForbidenError' instead of the 'MalformedUrlException'
        """
    AbiquoNodeDriver.connectionCls.conn_class = AbiquoMockHttp
    conn = AbiquoNodeDriver('son', 'gohan', 'http://dummy.host.com/api')
    self.assertRaises(ForbiddenError, conn.list_images)