import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_ECS
from libcloud.container.base import Container, ContainerImage, ContainerCluster
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.ecs import ElasticContainerDriver
from libcloud.container.utils.docker import RegistryClient
def test_list_containers_for_cluster(self):
    cluster = self.driver.list_clusters()[0]
    containers = self.driver.list_containers(cluster=cluster)
    self.assertEqual(len(containers), 1)