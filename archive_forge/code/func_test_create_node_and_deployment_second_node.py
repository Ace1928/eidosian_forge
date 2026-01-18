import os
import sys
import libcloud.security
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeAuthPassword
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
def test_create_node_and_deployment_second_node(self):
    kwargs = {'ex_storage_service_name': 'mtlytics', 'ex_deployment_name': 'dcoddkinztest02', 'ex_deployment_slot': 'Production', 'ex_admin_user_id': 'azurecoder'}
    auth = NodeAuthPassword('Pa55w0rd', False)
    kwargs['auth'] = auth
    kwargs['size'] = NodeSize(id='ExtraSmall', name='ExtraSmall', ram=1024, disk='30gb', bandwidth=0, price=0, driver=self.driver)
    kwargs['image'] = NodeImage(id='5112500ae3b842c8b9c604889f8753c3__OpenLogic-CentOS-65-20140415', name='FakeImage', driver=self.driver, extra={'vm_image': False})
    kwargs['name'] = 'dcoddkinztest03'
    result = self.driver.create_node(ex_cloud_service_name='testdcabc2', **kwargs)
    self.assertIsNotNone(result)