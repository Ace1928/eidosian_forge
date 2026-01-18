from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_delete_volume(self):
    v = cs.volumes.list()[0]
    del_v = v.delete()
    cs.assert_called('DELETE', '/volumes/1234')
    self._assert_request_id(del_v)
    del_v = cs.volumes.delete('1234')
    cs.assert_called('DELETE', '/volumes/1234')
    self._assert_request_id(del_v)
    del_v = cs.volumes.delete(v)
    cs.assert_called('DELETE', '/volumes/1234')
    self._assert_request_id(del_v)