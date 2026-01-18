import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@ddt.data({'target_info': {'target_lun': 1, 'target_wwn': '1234567890123456'}, 'expected_targets': [('1234567890123456', 1)]}, {'target_info': {'target_lun': 1, 'target_wwn': ['1234567890123456', '1234567890123457']}, 'expected_targets': [('1234567890123456', 1), ('1234567890123457', 1)]}, {'target_info': {'target_luns': [1, 1], 'target_wwn': ['1234567890123456', '1234567890123457']}, 'expected_targets': [('1234567890123456', 1), ('1234567890123457', 1)]}, {'target_info': {'target_luns': [1, 2], 'target_wwn': ['1234567890123456', '1234567890123457']}, 'expected_targets': [('1234567890123456', 1), ('1234567890123457', 2)]}, {'target_info': {'target_luns': [1, 1], 'target_wwns': ['1234567890123456', '1234567890123457']}, 'expected_targets': [('1234567890123456', 1), ('1234567890123457', 1)]}, {'target_info': {'target_lun': 7, 'target_luns': [1, 1], 'target_wwn': 'foo', 'target_wwns': ['1234567890123456', '1234567890123457']}, 'expected_targets': [('1234567890123456', 1), ('1234567890123457', 1)]}, {'target_info': {'target_lun': 1, 'target_wwn': '1234567890123456'}, 'expected_targets': [('1234567890123456', 1)], 'itmap': {'0004567890123456': ['1234567890123456']}, 'expected_map': {'0004567890123456': [('1234567890123456', 1)]}}, {'target_info': {'target_lun': 1, 'target_wwn': ['1234567890123456', '1234567890123457']}, 'expected_targets': [('1234567890123456', 1), ('1234567890123457', 1)], 'itmap': {'0004567890123456': ['1234567890123456', '1234567890123457']}, 'expected_map': {'0004567890123456': [('1234567890123456', 1), ('1234567890123457', 1)]}}, {'target_info': {'target_luns': [1, 2], 'target_wwn': ['1234567890123456', '1234567890123457']}, 'expected_targets': [('1234567890123456', 1), ('1234567890123457', 2)], 'itmap': {'0004567890123456': ['1234567890123456'], '1004567890123456': ['1234567890123457']}, 'expected_map': {'0004567890123456': [('1234567890123456', 1)], '1004567890123456': [('1234567890123457', 2)]}}, {'target_info': {'target_luns': [1, 2], 'target_wwn': ['1234567890123456', '1234567890123457']}, 'expected_targets': [('1234567890123456', 1), ('1234567890123457', 2)], 'itmap': {'0004567890123456': ['1234567890123456', '1234567890123457']}, 'expected_map': {'0004567890123456': [('1234567890123456', 1), ('1234567890123457', 2)]}}, {'target_info': {'target_lun': 1, 'target_wwn': ['20320002AC01E166', '21420002AC01E166', '20410002AC01E166', '21410002AC01E166']}, 'expected_targets': [('20320002ac01e166', 1), ('21420002ac01e166', 1), ('20410002ac01e166', 1), ('21410002ac01e166', 1)], 'itmap': {'10001409DCD71FF6': ['20320002AC01E166', '21420002AC01E166'], '10001409DCD71FF7': ['20410002AC01E166', '21410002AC01E166']}, 'expected_map': {'10001409dcd71ff6': [('20320002ac01e166', 1), ('21420002ac01e166', 1)], '10001409dcd71ff7': [('20410002ac01e166', 1), ('21410002ac01e166', 1)]}})
@ddt.unpack
def test__add_targets_to_connection_properties(self, target_info, expected_targets, itmap=None, expected_map=None):
    volume = {'id': 'fake_uuid'}
    wwn = '1234567890123456'
    conn = self.fibrechan_connection(volume, '10.0.2.15:3260', wwn)
    conn['data'].update(target_info)
    conn['data']['initiator_target_map'] = itmap
    connection_info = self.connector._add_targets_to_connection_properties(conn['data'])
    self.assertIn('targets', connection_info)
    self.assertEqual(expected_targets, connection_info['targets'])
    key = 'target_wwns' if 'target_wwns' in target_info else 'target_wwn'
    wwns = target_info.get(key)
    wwns = [wwns] if isinstance(wwns, str) else wwns
    wwns = [w.lower() for w in wwns]
    if wwns:
        self.assertEqual(wwns, conn['data'][key])
    if itmap:
        self.assertIn('initiator_target_lun_map', connection_info)
        self.assertEqual(expected_map, connection_info['initiator_target_lun_map'])