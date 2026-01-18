from unittest import mock
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine.resources.openstack.swift import container as swift_c
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
@mock.patch('swiftclient.client.Connection.post_account')
@mock.patch('swiftclient.client.Connection.put_container')
def test_account_headers(self, mock_put, mock_post):
    stack = utils.parse_stack(self.t)
    container_name = utils.PhysName(stack.name, 'test_resource')
    self._create_container(stack, definition_name='SwiftAccountMetadata')
    mock_put.assert_called_once_with(container_name, {})
    expected = {'X-Account-Meta-Temp-Url-Key': 'secret'}
    mock_post.assert_called_once_with(expected)