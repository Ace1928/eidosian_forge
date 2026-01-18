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
def test_exception_is_thrown_is_paramiko_is_not_available(self, mock_ssh_module, _):
    self.driver.features = {'create_node': ['password']}
    self.driver.create_node = Mock()
    self.driver.create_node.return_value = self.node
    mock_ssh_module.have_paramiko = False
    try:
        self.driver.deploy_node(deploy=Mock())
    except RuntimeError as e:
        self.assertTrue(str(e).find('paramiko is not installed') != -1)
    else:
        self.fail('Exception was not thrown')
    mock_ssh_module.have_paramiko = True
    node = self.driver.deploy_node(deploy=Mock())
    self.assertEqual(self.node.id, node.id)