from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_checks, 'registered_checks', {})
@mock.patch.object(_parser, 'LOG')
def test_no_handler(self, mock_log):
    result = _parser._parse_check('no:handler')
    self.assertIsInstance(result, _checks.FalseCheck)
    mock_log.error.assert_called()