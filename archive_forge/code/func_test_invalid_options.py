from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import validator
def test_invalid_options(self):
    self.assertRaises(RuntimeError, validator._validate, self.conf)