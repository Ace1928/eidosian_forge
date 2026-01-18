import io
import sys
import textwrap
from unittest import mock
import fixtures
from oslotest import base
import tempfile
import testscenarios
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_config import generator
from oslo_config import types
import yaml
def test_cleanup_opts_dup_opt(self):
    o = [('namespace1', [('group1', self.opts + [self.opts[0]])])]
    e = [('namespace1', [('group1', self.opts)])]
    self.assertEqual(e, generator._cleanup_opts(o))