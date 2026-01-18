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
def test_dictify_context_without_get_logging_values(self, mock_deprecate):
    ctxt = AlternativeRequestContext(user='user', tenant='tenant')
    d = {'user': 'user', 'tenant': 'tenant'}
    self.assertEqual(d, formatters._dictify_context(ctxt))
    mock_deprecate.assert_called_with('The RequestContext.get_logging_values() method should be defined for logging context specific information.  The to_dict() method is deprecated for oslo.log use.', removal_version='5.0.0', version='3.8.0')