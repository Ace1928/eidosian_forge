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
@mock.patch.object(generator, '_get_raw_opts_loaders')
def test_list_ignores_doubles(self, raw_opts_loaders):
    config_opts = [(None, [cfg.StrOpt('foo'), cfg.StrOpt('bar')])]
    raw_opts_loaders.return_value = [('namespace', lambda: config_opts), ('namespace', lambda: config_opts)]
    slurped_opts = 0
    for _, listing in generator._list_opts(['namespace']):
        for _, opts in listing:
            slurped_opts += len(opts)
    self.assertEqual(2, slurped_opts)