import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_DOCKER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.docker import DockerContainerDriver
def test_ex_search_images(self):
    for driver in self.drivers:
        images = driver.ex_search_images('mysql')
        self.assertEqual(len(images), 25)
        self.assertEqual(images[0].name, 'mysql')