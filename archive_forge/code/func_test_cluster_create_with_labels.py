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
def test_cluster_create_with_labels(self):
    """Verifies labels are properly parsed when given as argument."""
    expected_args = self._default_args
    expected_args['labels'] = {'arg1': 'value1', 'arg2': 'value2'}
    arglist = ['--cluster-template', self._cluster.cluster_template_id, '--labels', 'arg1=value1', '--labels', 'arg2=value2', self._cluster.name]
    verifylist = [('cluster_template', self._cluster.cluster_template_id), ('labels', ['arg1=value1', 'arg2=value2']), ('name', self._cluster.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.clusters_mock.create.assert_called_with(**expected_args)