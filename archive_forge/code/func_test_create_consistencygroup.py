from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_consistencygroup(self):
    vol = cs.consistencygroups.create('type1,type2', 'cg')
    cs.assert_called('POST', '/consistencygroups')
    self._assert_request_id(vol)