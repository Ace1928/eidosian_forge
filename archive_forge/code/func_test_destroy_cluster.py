import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_ECS
from libcloud.container.base import Container, ContainerImage, ContainerCluster
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.ecs import ElasticContainerDriver
from libcloud.container.utils.docker import RegistryClient
def test_destroy_cluster(self):
    self.assertTrue(self.driver.destroy_cluster(ContainerCluster(id='arn:aws:ecs:us-east-1:012345678910:cluster/jim', name='jim', driver=self.driver)))