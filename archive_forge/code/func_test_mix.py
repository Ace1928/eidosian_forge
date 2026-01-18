import os
import tempfile
import textwrap
from openstack.config import loader
from openstack import exceptions
from openstack.tests.unit.config import base
def test_mix(self):
    argv = ['-a', '-b', '--long-arg', '--multi_value', 'key1=value1', '--multi-value', 'key2=value2']
    self.assertRaises(exceptions.ConfigException, loader._fix_argv, argv)