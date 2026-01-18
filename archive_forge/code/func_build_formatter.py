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
def build_formatter(output_file, **kwargs):
    conf = cfg.ConfigOpts()
    conf.register_opts(generator._generator_opts)
    for k, v in kwargs.items():
        conf.set_override(k, v)
    return generator._OptFormatter(conf, output_file=output_file)