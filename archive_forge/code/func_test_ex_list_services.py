import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def test_ex_list_services(self):
    services = self.driver.ex_list_services()
    self.assertEqual(len(services), 4)
    self.assertEqual(services[0]['id'], '1s1')