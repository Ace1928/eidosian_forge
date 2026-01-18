from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_get_associations(self):
    qos_id = '1B6B6A04-A927-4AEB-810B-B7BAAD49F57C'
    qos = cs.qos_specs.get_associations(qos_id)
    cs.assert_called('GET', '/qos-specs/%s/associations' % qos_id)
    self._assert_request_id(qos)