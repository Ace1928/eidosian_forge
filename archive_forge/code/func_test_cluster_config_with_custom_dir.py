import copy
import os
import sys
import tempfile
from unittest import mock
from contextlib import contextmanager
from unittest.mock import call
from magnumclient import exceptions
from magnumclient.osc.v1 import clusters as osc_clusters
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
@mock.patch.dict(os.environ, {'SHELL': '/bin/bash'})
def test_cluster_config_with_custom_dir(self):
    tmp_dir = tempfile.mkdtemp()
    arglist = ['fake-cluster', '--dir', tmp_dir]
    verifylist = [('cluster', 'fake-cluster'), ('dir', tmp_dir)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    expected_value = 'export KUBECONFIG={}/config\n\n'.format(tmp_dir)
    with capture(self.cmd.take_action, parsed_args) as output:
        self.assertEqual(expected_value, output)
    self.clusters_mock.get.assert_called_with('fake-cluster')