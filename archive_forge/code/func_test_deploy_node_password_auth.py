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
def test_deploy_node_password_auth(self, mock_ssh_module, _):
    self.driver.features = {'create_node': ['password']}
    mock_ssh_module.have_paramiko = True
    self.driver.create_node = Mock()
    self.driver.create_node.return_value = self.node
    node = self.driver.deploy_node(deploy=Mock())
    self.assertEqual(self.node.id, node.id)