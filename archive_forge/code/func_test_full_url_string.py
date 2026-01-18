import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def test_full_url_string(self):
    """
        Test a 'full' URL string, which contains a scheme, port, and base path.
        """
    path = 'http://myhostname:1234/base'
    driver = RancherContainerDriver(*CONTAINER_PARAMS_RANCHER, host=path)
    self.assertEqual(driver.secure, False)
    self.assertEqual(driver.connection.host, 'myhostname')
    self.assertEqual(driver.connection.port, 1234)
    self.assertEqual(driver.baseuri, '/base')