import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_list_deployments(self):
    deployments = self.driver.ex_list_deployments()
    self.assertEqual(len(deployments), 7)
    deployment = deployments[0]
    self.assertEqual(deployment.id, 'aea45586-9a4a-4a01-805c-719f431316c0')
    self.assertEqual(deployment.name, 'event-exporter-gke')
    self.assertEqual(deployment.namespace, 'kube-system')
    for deployment in deployments:
        self.assertIsInstance(deployment.replicas, int)
        self.assertIsInstance(deployment.selector, dict)