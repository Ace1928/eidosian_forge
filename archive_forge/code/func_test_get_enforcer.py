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
def test_get_enforcer(self, mock_manager):
    mock_instance = mock.MagicMock()
    mock_instance.__contains__.return_value = True
    mock_manager.return_value = mock_instance
    mock_item = mock.Mock()
    mock_item.obj = 'test'
    mock_instance.__getitem__.return_value = mock_item
    self.assertEqual('test', generator._get_enforcer('foo'))