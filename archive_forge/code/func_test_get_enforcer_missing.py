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
def test_get_enforcer_missing(self, mock_manager):
    mock_instance = mock.MagicMock()
    mock_instance.__contains__.return_value = False
    mock_manager.return_value = mock_instance
    self.assertRaises(KeyError, generator._get_enforcer, 'nonexistent')