from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_mclag_attribute_requests(self, domain_id, command):
    requests = []
    url_common = 'data/openconfig-mclag:mclag/mclag-domains/mclag-domain=%s/config' % domain_id
    method = DELETE
    if 'source_address' in command and command['source_address'] is not None:
        url = url_common + '/source-address'
        request = {'path': url, 'method': method}
        requests.append(request)
    if 'peer_address' in command and command['peer_address'] is not None:
        url = url_common + '/peer-address'
        request = {'path': url, 'method': method}
        requests.append(request)
    if 'peer_link' in command and command['peer_link'] is not None:
        url = url_common + '/peer-link'
        request = {'path': url, 'method': method}
        requests.append(request)
    if 'keepalive' in command and command['keepalive'] is not None:
        url = url_common + '/keepalive-interval'
        request = {'path': url, 'method': method}
        requests.append(request)
    if 'session_timeout' in command and command['session_timeout'] is not None:
        url = url_common + '/session-timeout'
        request = {'path': url, 'method': method}
        requests.append(request)
    if 'system_mac' in command and command['system_mac'] is not None:
        url = url_common + '/mclag-system-mac'
        request = {'path': url, 'method': method}
        requests.append(request)
    if 'delay_restore' in command and command['delay_restore'] is not None:
        url = url_common + '/delay-restore'
        request = {'path': url, 'method': method}
        requests.append(request)
    if 'peer_gateway' in command and command['peer_gateway'] is not None:
        if command['peer_gateway']['vlans'] is None:
            request = {'path': 'data/openconfig-mclag:mclag/vlan-ifs/vlan-if', 'method': method}
            requests.append(request)
        elif command['peer_gateway']['vlans'] is not None:
            vlan_id_list = self.get_vlan_id_list(command['peer_gateway']['vlans'])
            for vlan in vlan_id_list:
                peer_gateway_url = 'data/openconfig-mclag:mclag/vlan-ifs/vlan-if=Vlan{0}'.format(vlan)
                request = {'path': peer_gateway_url, 'method': method}
                requests.append(request)
    if 'unique_ip' in command and command['unique_ip'] is not None:
        if command['unique_ip']['vlans'] is None:
            request = {'path': 'data/openconfig-mclag:mclag/vlan-interfaces/vlan-interface', 'method': method}
            requests.append(request)
        elif command['unique_ip']['vlans'] is not None:
            vlan_id_list = self.get_vlan_id_list(command['unique_ip']['vlans'])
            for vlan in vlan_id_list:
                unique_ip_url = 'data/openconfig-mclag:mclag/vlan-interfaces/vlan-interface=Vlan{0}'.format(vlan)
                request = {'path': unique_ip_url, 'method': method}
                requests.append(request)
    if 'members' in command and command['members'] is not None:
        if command['members']['portchannels'] is None:
            request = {'path': 'data/openconfig-mclag:mclag/interfaces/interface', 'method': method}
            requests.append(request)
        elif command['members']['portchannels'] is not None:
            for each in command['members']['portchannels']:
                if each:
                    portchannel_url = 'data/openconfig-mclag:mclag/interfaces/interface=%s' % each['lag']
                    request = {'path': portchannel_url, 'method': method}
                    requests.append(request)
    if 'gateway_mac' in command and command['gateway_mac'] is not None:
        request = {'path': 'data/openconfig-mclag:mclag/mclag-gateway-macs/mclag-gateway-mac', 'method': method}
        requests.append(request)
    return requests