from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecated_message(self, mock_reporter):
    versionutils.deprecation_warning('outdated_stuff', as_of=versionutils.deprecated.KILO, in_favor_of='different_stuff', remove_in=+2)
    self.assert_deprecated(mock_reporter, what='outdated_stuff', in_favor_of='different_stuff', as_of='Kilo', remove_in='Mitaka')