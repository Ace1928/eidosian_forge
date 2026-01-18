from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecated_class_without_init(self, mock_reporter):

    @versionutils.deprecated(as_of=versionutils.deprecated.JUNO, remove_in=+1)
    class OutdatedClass(object):
        pass
    obj = OutdatedClass()
    self.assertIsInstance(obj, OutdatedClass)
    self.assert_deprecated(mock_reporter, what='OutdatedClass()', as_of='Juno', remove_in='Kilo')