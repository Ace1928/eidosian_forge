import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_creating_node_from_cdrom_image(self):
    image = NodeImage(id='01000000-0000-4000-8000-000030060200', name='Ubuntu Server 16.04 LTS (Xenial Xerus)', driver='', extra={'type': 'cdrom'})
    body = UpcloudCreateNodeRequestBody(name='ts', image=image, location=self.location, size=self.size)
    json_body = body.to_json()
    dict_body = json.loads(json_body)
    expected_body = {'server': {'title': 'ts', 'hostname': 'localhost', 'plan': '1xCPU-1GB', 'zone': 'fi-hel1', 'login_user': {'username': 'root', 'create_password': 'yes'}, 'storage_devices': {'storage_device': [{'action': 'create', 'size': 30, 'tier': 'maxiops', 'title': 'Ubuntu Server 16.04 LTS (Xenial Xerus)'}, {'action': 'attach', 'storage': '01000000-0000-4000-8000-000030060200', 'type': 'cdrom'}]}}}
    self.assertDictEqual(expected_body, dict_body)