from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_ntp(delta, existing):
    if delta.get('address') or delta.get('peer_type') or delta.get('vrf_name') or delta.get('key_id') or delta.get('prefer'):
        address = delta.get('address', existing.get('address'))
        peer_type = delta.get('peer_type', existing.get('peer_type'))
        key_id = delta.get('key_id', existing.get('key_id'))
        prefer = delta.get('prefer', existing.get('prefer'))
        vrf_name = delta.get('vrf_name', existing.get('vrf_name'))
        if delta.get('key_id') == 'default':
            key_id = None
    else:
        peer_type = None
        prefer = None
    source_type = delta.get('source_type')
    source = delta.get('source')
    if prefer:
        if prefer == 'enabled':
            prefer = True
        elif prefer == 'disabled':
            prefer = False
    if source:
        source_type = delta.get('source_type', existing.get('source_type'))
    ntp_cmds = []
    if peer_type:
        if existing.get('peer_type') and existing.get('address'):
            ntp_cmds.append('no ntp {0} {1}'.format(existing.get('peer_type'), existing.get('address')))
        ntp_cmds.append(set_ntp_server_peer(peer_type, address, prefer, key_id, vrf_name))
    if source:
        existing_source_type = existing.get('source_type')
        existing_source = existing.get('source')
        if existing_source_type and source_type != existing_source_type:
            ntp_cmds.append('no ntp {0} {1}'.format(existing_source_type, existing_source))
        if source == 'default':
            if existing_source_type and existing_source:
                ntp_cmds.append('no ntp {0} {1}'.format(existing_source_type, existing_source))
        else:
            ntp_cmds.append('ntp {0} {1}'.format(source_type, source))
    return ntp_cmds