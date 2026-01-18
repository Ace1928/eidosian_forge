from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecated_exception_new(self, mock_log):

    @versionutils.deprecated(as_of=versionutils.deprecated.ICEHOUSE, remove_in=+1)
    class OldException(Exception):
        pass

    class NewException(OldException):
        pass
    try:
        raise NewException()
    except NewException:
        pass
    mock_log.assert_not_called()