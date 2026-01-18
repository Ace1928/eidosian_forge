from unittest import mock
import yaml
from heat.common import environment_format
from heat.tests import common
def test_bad_yaml(self):
    env = '\nparameters: }\n'
    self.assertRaises(ValueError, environment_format.parse, env)