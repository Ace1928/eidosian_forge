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
def test_cleanup_opts_dup_mixed_case_groups_opt(self):
    o = [('namespace1', [('default', self.opts), ('Default', self.opts + [self.opts[1]]), ('DEFAULT', self.opts + [self.opts[2]]), ('group1', self.opts + [self.opts[1]]), ('Group1', self.opts), ('GROUP1', self.opts + [self.opts[2]])])]
    e = [('namespace1', [('DEFAULT', self.opts), ('group1', self.opts)])]
    self.assertEqual(e, generator._cleanup_opts(o))