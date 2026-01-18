import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_creating_node_with_non_default_username(self):
    body = UpcloudCreateNodeRequestBody(name='ts', image=self.image, location=self.location, size=self.size, ex_username='someone')
    json_body = body.to_json()
    dict_body = json.loads(json_body)
    login_user = dict_body['server']['login_user']
    self.assertDictEqual({'username': 'someone', 'create_password': 'yes'}, login_user)