import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def test_ex_deploy_stack(self):
    stack = self.driver.ex_deploy_stack(name='newstack', environment={'root_password': 'password'})
    self.assertEqual(stack['id'], '1e9')
    self.assertEqual(stack['environment']['root_password'], 'password')