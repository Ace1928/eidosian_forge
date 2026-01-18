import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
@ddt.data({'do_raise': False, 'force': False}, {'do_raise': True, 'force': True}, {'do_raise': True, 'force': False})
@ddt.unpack
@mock.patch.object(iscsi.ISCSIConnector, '_disconnect_from_iscsi_portal')
def test_disconnect_connection(self, disconnect_mock, do_raise, force):
    will_raise = do_raise and (not force)
    actual_call_args = []

    def my_disconnect(con_props):
        actual_call_args.append(con_props.copy())
        if do_raise:
            raise exception.ExceptionChainer()
    disconnect_mock.side_effect = my_disconnect
    connections = (('ip1:port1', 'tgt1'), ('ip2:port2', 'tgt2'))
    original_props = self.CON_PROPS.copy()
    exc = exception.ExceptionChainer()
    if will_raise:
        self.assertRaises(exception.ExceptionChainer, self.connector._disconnect_connection, self.CON_PROPS, connections, force=force, exc=exc)
    else:
        self.connector._disconnect_connection(self.CON_PROPS, connections, force=force, exc=exc)
    self.assertDictEqual(original_props, self.CON_PROPS)
    expected = [original_props.copy(), original_props.copy()]
    for i, (ip, iqn) in enumerate(connections):
        expected[i].update(target_portal=ip, target_iqn=iqn)
    if will_raise:
        expected = expected[:1]
    self.assertListEqual(expected, actual_call_args)
    self.assertEqual(do_raise, bool(exc))