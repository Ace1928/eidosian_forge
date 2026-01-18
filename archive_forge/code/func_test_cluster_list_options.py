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
def test_cluster_list_options(self):
    arglist = ['--limit', '1', '--sort-key', 'key', '--sort-dir', 'asc']
    verifylist = [('limit', 1), ('sort_key', 'key'), ('sort_dir', 'asc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.clusters_mock.list.assert_called_with(limit=1, sort_dir='asc', sort_key='key')