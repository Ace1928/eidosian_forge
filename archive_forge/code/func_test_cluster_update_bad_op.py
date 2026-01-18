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
def test_cluster_update_bad_op(self):
    arglist = ['foo', 'bar', 'snafu']
    verifylist = [('cluster', 'foo'), ('op', 'bar'), ('attributes', ['snafu']), ('rollback', False)]
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)