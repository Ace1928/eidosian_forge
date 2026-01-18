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
@patch('libcloud.compute.base.atexit')
@patch('libcloud.compute.base.SSHClient')
@patch('libcloud.compute.ssh')
def test_deploy_node_at_exit_func_functionality(self, mock_ssh_module, _, mock_at_exit):
    self.driver.create_node = Mock()
    self.driver.create_node.return_value = self.node
    mock_ssh_module.have_paramiko = True
    deploy = Mock()

    def mock_at_exit_func(driver, node):
        pass
    self.assertEqual(mock_at_exit.register.call_count, 0)
    self.assertEqual(mock_at_exit.unregister.call_count, 0)
    node = self.driver.deploy_node(deploy=deploy, at_exit_func=mock_at_exit_func)
    self.assertEqual(mock_at_exit.register.call_count, 1)
    self.assertEqual(mock_at_exit.unregister.call_count, 1)
    self.assertEqual(self.node.id, node.id)
    mock_at_exit.reset_mock()
    deploy.run.side_effect = Exception('foo')
    self.assertEqual(mock_at_exit.register.call_count, 0)
    self.assertEqual(mock_at_exit.unregister.call_count, 0)
    try:
        self.driver.deploy_node(deploy=deploy, at_exit_func=mock_at_exit_func)
    except DeploymentError as e:
        self.assertTrue(e.node.id, self.node.id)
    else:
        self.fail('Exception was not thrown')
    self.assertEqual(mock_at_exit.register.call_count, 1)
    self.assertEqual(mock_at_exit.unregister.call_count, 1)
    mock_at_exit.reset_mock()
    self.driver.create_node = Mock(side_effect=Exception('Failure'))
    self.assertEqual(mock_at_exit.register.call_count, 0)
    self.assertEqual(mock_at_exit.unregister.call_count, 0)
    try:
        self.driver.deploy_node(deploy=deploy, at_exit_func=mock_at_exit_func)
    except Exception as e:
        self.assertTrue('Failure' in str(e))
    else:
        self.fail('Exception was not thrown')
    self.assertEqual(mock_at_exit.register.call_count, 0)
    self.assertEqual(mock_at_exit.unregister.call_count, 0)