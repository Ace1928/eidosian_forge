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
@patch('libcloud.compute.ssh')
def test_deploy_node_depoy_node_not_implemented(self, mock_ssh_module):
    self.driver.features = {'create_node': []}
    mock_ssh_module.have_paramiko = True
    try:
        self.driver.deploy_node(deploy=Mock())
    except NotImplementedError:
        pass
    else:
        self.fail('Exception was not thrown')
    self.driver.features = {}
    try:
        self.driver.deploy_node(deploy=Mock())
    except NotImplementedError:
        pass
    else:
        self.fail('Exception was not thrown')