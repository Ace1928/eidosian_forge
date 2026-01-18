import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_list_nodes_metrics(self):
    nodes_metrics = self.driver.ex_list_nodes_metrics()
    self.assertEqual(len(nodes_metrics), 1)
    self.assertEqual(nodes_metrics[0]['metadata']['name'], 'gke-cluster-3-default-pool-76fd57f7-l83v')