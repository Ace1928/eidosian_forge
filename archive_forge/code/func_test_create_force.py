from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_force(self):
    vol = cs.backups.create('2b695faf-b963-40c8-8464-274008fbcef4', None, None, False, True)
    cs.assert_called('POST', '/backups')
    self._assert_request_id(vol)