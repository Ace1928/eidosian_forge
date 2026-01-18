from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
def assert_deprecated(self, mock_reporter, no_removal=False, **expected_details):
    if 'in_favor_of' in expected_details:
        if no_removal is False:
            expected_msg = versionutils._deprecated_msg_with_alternative
        else:
            expected_msg = getattr(versionutils, '_deprecated_msg_with_alternative_no_removal')
    elif no_removal is False:
        expected_msg = versionutils._deprecated_msg_no_alternative
    else:
        expected_msg = getattr(versionutils, '_deprecated_msg_with_no_alternative_no_removal')
    mock_reporter.assert_called_with(mock.ANY, expected_msg, expected_details)