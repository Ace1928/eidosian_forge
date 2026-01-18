import logging
import sys
from unittest import mock
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
from oslotest import base as test_base
from oslo_log import formatters
from oslo_log import log
@mock.patch('debtcollector.deprecate')
def test_dictify_context_with_dict(self, mock_deprecate):
    d = {'user': 'user'}
    self.assertEqual(d, formatters._dictify_context(d))
    mock_deprecate.assert_not_called()