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
def test_get_groups_empty_ns(self):
    groups = generator._get_groups([])
    self.assertEqual({'DEFAULT': {'object': None, 'namespaces': []}}, groups)