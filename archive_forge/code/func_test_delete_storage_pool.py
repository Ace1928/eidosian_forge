import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
def test_delete_storage_pool(self):
    for driver in self.drivers:
        driver.ex_delete_storage_pool(id='pool1')