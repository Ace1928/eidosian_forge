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
def test_dictify_context_with_context(self, mock_deprecate):
    ctxt = _fake_context()
    self.assertEqual(ctxt.get_logging_values(), formatters._dictify_context(ctxt))
    mock_deprecate.assert_not_called()