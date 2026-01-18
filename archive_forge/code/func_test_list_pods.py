import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_list_pods(self):
    pods = self.driver.ex_list_pods()
    self.assertEqual(len(pods), 1)
    self.assertEqual(pods[0].id, '1fad5411-b9af-11e5-8701-0050568157ec')
    self.assertEqual(pods[0].name, 'hello-world')