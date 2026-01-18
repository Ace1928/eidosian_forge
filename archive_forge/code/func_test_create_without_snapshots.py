from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_without_snapshots(self):
    vol = v355cs.transfers.create('1234', no_snapshots=True)
    v355cs.assert_called('POST', '/%s' % TRANSFER_355_URL, body={'transfer': {'volume_id': '1234', 'name': None, 'no_snapshots': True}})
    self._assert_request_id(vol)