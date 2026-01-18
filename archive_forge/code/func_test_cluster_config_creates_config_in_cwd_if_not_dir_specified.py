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
def test_cluster_config_creates_config_in_cwd_if_not_dir_specified(self):
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)
    arglist = ['fake-cluster']
    verifylist = [('cluster', 'fake-cluster')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    expected_value = 'export KUBECONFIG={}/config\n\n'.format(os.getcwd())
    with capture(self.cmd.take_action, parsed_args) as output:
        self.assertEqual(expected_value, output)
    self.clusters_mock.get.assert_called_with('fake-cluster')