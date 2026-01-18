from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_callback_priority(self):
    pri_first = priority_group.PRIORITY_DEFAULT - 100
    pri_last = priority_group.PRIORITY_DEFAULT + 100
    self.manager.subscribe(callback_1, 'my-resource', 'my-event')
    self.manager.subscribe(callback_2, 'my-resource', 'my-event', pri_last)
    self.manager.subscribe(callback_3, 'my-resource', 'my-event', pri_first)
    callbacks = self.manager._callbacks['my-resource']['my-event']
    self.assertEqual(3, len(callbacks))
    self.assertEqual(pri_first, callbacks[0][0])
    self.assertEqual(priority_group.PRIORITY_DEFAULT, callbacks[1][0])
    self.assertEqual(pri_last, callbacks[2][0])