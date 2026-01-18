import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
def test_generator_call_with_no_arguments_raises_error(self):
    testargs = ['oslopolicy-sample-generator']
    with mock.patch('sys.argv', testargs):
        local_conf = cfg.ConfigOpts()
        self.assertRaises(cfg.RequiredOptError, generator.generate_sample, [], local_conf)