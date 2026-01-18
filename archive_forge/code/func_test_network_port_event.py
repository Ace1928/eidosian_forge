import testtools
from ironicclient.tests.unit import utils
from ironicclient.v1 import events
def test_network_port_event(self):
    evts = self.mgr.create(**FAKE_NETWORK_PORT_EVENTS)
    expect = [('POST', '/v1/events', {}, FAKE_NETWORK_PORT_EVENTS)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(evts)