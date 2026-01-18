from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
def test_decorated_inst_method_receives(self):
    i1 = ObjectWithDecoratedCallback()
    event_payload = events.EventPayload(mock.ANY)
    registry.publish(resources.PORT, events.BEFORE_CREATE, self, payload=event_payload)
    self.assertEqual(0, i1.counter)
    registry.publish(resources.PORT, events.AFTER_CREATE, self, payload=event_payload)
    self.assertEqual(1, i1.counter)
    registry.publish(resources.PORT, events.AFTER_UPDATE, self, payload=event_payload)
    self.assertEqual(2, i1.counter)
    registry.publish(resources.NETWORK, events.AFTER_UPDATE, self, payload=event_payload)
    self.assertEqual(2, i1.counter)
    registry.publish(resources.NETWORK, events.AFTER_DELETE, self, payload=event_payload)
    self.assertEqual(3, i1.counter)
    i2 = ObjectWithDecoratedCallback()
    self.assertEqual(0, i2.counter)
    registry.publish(resources.NETWORK, events.AFTER_DELETE, self, payload=event_payload)
    self.assertEqual(4, i1.counter)
    self.assertEqual(1, i2.counter)