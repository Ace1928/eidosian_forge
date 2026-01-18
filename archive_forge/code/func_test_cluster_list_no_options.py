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
def test_cluster_list_no_options(self):
    arglist = []
    verifylist = [('limit', None), ('sort_key', None), ('sort_dir', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.clusters_mock.list.assert_called_with(limit=None, sort_dir=None, sort_key=None)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))