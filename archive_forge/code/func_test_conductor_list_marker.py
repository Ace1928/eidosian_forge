import testtools
from testtools.matchers import HasLength
from ironicclient.tests.unit import utils
from ironicclient.v1 import conductor
def test_conductor_list_marker(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = conductor.ConductorManager(self.api)
    conductors = self.mgr.list(marker=CONDUCTOR1['hostname'])
    expect = [('GET', '/v1/conductors/?marker=%s' % CONDUCTOR1['hostname'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(conductors, HasLength(1))