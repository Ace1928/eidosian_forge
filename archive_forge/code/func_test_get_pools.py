from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_get_pools(self):
    vol = cs.volumes.get_pools('')
    cs.assert_called('GET', '/scheduler-stats/get_pools')
    self._assert_request_id(vol)