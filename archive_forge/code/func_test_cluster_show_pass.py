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
def test_cluster_show_pass(self):
    arglist = ['fake-cluster']
    verifylist = [('cluster', 'fake-cluster')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.clusters_mock.get.assert_called_with('fake-cluster')
    self.assertEqual(osc_clusters.CLUSTER_ATTRIBUTES, columns)
    self.assertEqual(self.data, data)