from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_attach_to_host(self):
    v = cs.volumes.get('1234')
    self._assert_request_id(v)
    vol = cs.volumes.attach(v, None, None, host_name='test', mode='rw')
    cs.assert_called('POST', '/volumes/1234/action')
    self._assert_request_id(vol)