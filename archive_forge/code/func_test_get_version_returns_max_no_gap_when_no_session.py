from unittest import mock
import oslotest.base as base
from osc_placement import version
def test_get_version_returns_max_no_gap_when_no_session(self):
    obj = mock.Mock()
    obj.app.client_manager.session = None
    ret = version.get_version(obj)
    self.assertEqual(version.MAX_VERSION_NO_GAP, ret)
    obj.app.client_manager.placement.api_version.assert_not_called()