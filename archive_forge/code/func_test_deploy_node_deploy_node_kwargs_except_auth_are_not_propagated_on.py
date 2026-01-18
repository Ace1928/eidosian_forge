import os
import sys
import time
import unittest
from unittest.mock import Mock, patch
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.utils.py3 import u, httplib, assertRaisesRegex
from libcloud.compute.ssh import BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.compute.base import Node, NodeAuthPassword
from libcloud.test.secrets import RACKSPACE_PARAMS
from libcloud.compute.types import NodeState, LibcloudError, DeploymentError
from libcloud.compute.deployment import (
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.rackspace import RackspaceFirstGenNodeDriver as Rackspace
@patch('libcloud.compute.base.SSHClient')
@patch('libcloud.compute.ssh')
def test_deploy_node_deploy_node_kwargs_except_auth_are_not_propagated_on(self, mock_ssh_module, _):
    mock_ssh_module.have_paramiko = True
    self.driver.create_node = Mock()
    self.driver.create_node.return_value = self.node
    self.driver._connect_and_run_deployment_script = Mock()
    self.driver._wait_until_running = Mock()
    kwargs = {}
    for key in DEPLOY_NODE_KWARGS:
        kwargs[key] = key
    kwargs['ssh_interface'] = 'public_ips'
    kwargs['ssh_alternate_usernames'] = ['foo', 'bar']
    kwargs['timeout'] = 10
    auth = NodeAuthPassword('P@$$w0rd')
    node = self.driver.deploy_node(name='name', image='image', size='size', auth=auth, ex_foo='ex_foo', **kwargs)
    self.assertEqual(self.node.id, node.id)
    self.assertEqual(self.driver.create_node.call_count, 1)
    call_kwargs = self.driver.create_node.call_args_list[0][1]
    expected_call_kwargs = {'name': 'name', 'image': 'image', 'size': 'size', 'auth': auth, 'ex_foo': 'ex_foo'}
    self.assertEqual(expected_call_kwargs, call_kwargs)
    global call_count
    call_count = 0

    def create_node(name, image, size, ex_custom_arg_1, ex_custom_arg_2, ex_foo=None, auth=None, **kwargs):
        global call_count
        call_count += 1
        if call_count == 1:
            msg = 'create_node() takes at least 5 arguments (7 given)'
            raise TypeError(msg)
        return self.node
    self.driver.create_node = create_node
    node = self.driver.deploy_node(name='name', image='image', size='size', auth=auth, ex_foo='ex_foo', ex_custom_arg_1='a', ex_custom_arg_2='b', **kwargs)
    self.assertEqual(self.node.id, node.id)
    self.assertEqual(call_count, 2)
    call_count = 0

    def create_node(name, image, size, ex_custom_arg_1, ex_custom_arg_2, ex_foo=None, auth=None, **kwargs):
        global call_count
        call_count += 1
        if call_count == 1:
            msg = 'create_node() missing 3 required positional arguments'
            raise TypeError(msg)
        return self.node
    self.driver.create_node = create_node
    node = self.driver.deploy_node(name='name', image='image', size='size', auth=auth, ex_foo='ex_foo', ex_custom_arg_1='a', ex_custom_arg_2='b', **kwargs)
    self.assertEqual(self.node.id, node.id)
    self.assertEqual(call_count, 2)