import testtools
from testtools.matchers import HasLength
from ironicclient.tests.unit import utils
from ironicclient.v1 import conductor
def test_conductor_list_sort_key(self):
    self.api = utils.FakeAPI(fake_responses_sorting)
    self.mgr = conductor.ConductorManager(self.api)
    conductors = self.mgr.list(sort_key='updated_at')
    expect = [('GET', '/v1/conductors/?sort_key=updated_at', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(conductors))