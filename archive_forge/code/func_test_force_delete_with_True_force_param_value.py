from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_force_delete_with_True_force_param_value(self):
    """Tests delete backup with force parameter set to True"""
    b = cs.backups.list()[0]
    del_back = b.delete(force=True)
    expected_body = {'os-force_delete': None}
    cs.assert_called('POST', '/backups/76a17945-3c6f-435c-975b-b5685db10b62/action', expected_body)
    self._assert_request_id(del_back)