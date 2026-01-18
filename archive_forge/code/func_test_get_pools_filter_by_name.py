from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@ddt.data(True, False)
def test_get_pools_filter_by_name(self, detail):
    cs = fakes.FakeClient(api_version=api_versions.APIVersion('3.33'))
    vol = cs.volumes.get_pools(detail, {'name': 'pool1'})
    request_url = '/scheduler-stats/get_pools?name=pool1'
    if detail:
        request_url = '/scheduler-stats/get_pools?detail=True&name=pool1'
    cs.assert_called('GET', request_url)
    self._assert_request_id(vol)