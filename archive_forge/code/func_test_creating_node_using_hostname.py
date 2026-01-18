import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_creating_node_using_hostname(self):
    body = UpcloudCreateNodeRequestBody(name='ts', image=self.image, location=self.location, size=self.size, ex_hostname='myhost.upcloud.com')
    json_body = body.to_json()
    dict_body = json.loads(json_body)
    expected_body = {'server': {'title': 'ts', 'hostname': 'myhost.upcloud.com', 'plan': '1xCPU-1GB', 'zone': 'fi-hel1', 'login_user': {'username': 'root', 'create_password': 'yes'}, 'storage_devices': {'storage_device': [{'action': 'clone', 'title': 'Ubuntu Server 16.04 LTS (Xenial Xerus)', 'storage': '01000000-0000-4000-8000-000030060200', 'tier': 'maxiops', 'size': 30}]}}}
    self.assertDictEqual(expected_body, dict_body)