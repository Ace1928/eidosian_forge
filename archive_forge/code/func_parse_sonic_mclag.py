from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.mclag.mclag import MclagArgs
from ansible.module_utils.connection import ConnectionError
def parse_sonic_mclag(self, spec, conf):
    config = {}
    portchannels_list = []
    if conf:
        domain_data = None
        if conf.get('mclag-domains', None) and conf['mclag-domains'].get('mclag-domain', None):
            domain_data = conf['mclag-domains']['mclag-domain'][0]
        if domain_data:
            domain_id = domain_data['domain-id']
            config['domain_id'] = domain_id
            domain_config = domain_data.get('config', None)
            if domain_config:
                if domain_config.get('session-timeout', None):
                    config['session_timeout'] = domain_config['session-timeout']
                if domain_config.get('keepalive-interval', None):
                    config['keepalive'] = domain_config['keepalive-interval']
                if domain_config.get('source-address', None):
                    config['source_address'] = domain_config['source-address']
                if domain_config.get('peer-address', None):
                    config['peer_address'] = domain_config['peer-address']
                if domain_config.get('peer-link', None):
                    config['peer_link'] = domain_config['peer-link']
                if domain_config.get('mclag-system-mac', None):
                    config['system_mac'] = domain_config['mclag-system-mac']
                if domain_config.get('delay-restore', None):
                    config['delay_restore'] = domain_config['delay-restore']
            if conf.get('vlan-interfaces', None) and conf['vlan-interfaces'].get('vlan-interface', None):
                vlans_list = []
                vlan_data = conf['vlan-interfaces']['vlan-interface']
                for vlan in vlan_data:
                    vlans_list.append({'vlan': vlan['name']})
                if vlans_list:
                    config['unique_ip'] = {'vlans': self.get_vlan_range_list(vlans_list)}
            if conf.get('vlan-ifs', None) and conf['vlan-ifs'].get('vlan-if', None):
                vlans_list = []
                vlan_data = conf['vlan-ifs']['vlan-if']
                for vlan in vlan_data:
                    vlans_list.append({'vlan': vlan['name']})
                if vlans_list:
                    config['peer_gateway'] = {'vlans': self.get_vlan_range_list(vlans_list)}
            if conf.get('interfaces', None) and conf['interfaces'].get('interface', None):
                portchannels_list = []
                po_data = conf['interfaces']['interface']
                for po in po_data:
                    if po.get('config', None) and po['config'].get('mclag-domain-id', None) and (domain_id == domain_data['domain-id']):
                        portchannels_list.append({'lag': po['name']})
                if portchannels_list:
                    config['members'] = {'portchannels': portchannels_list}
            if conf.get('mclag-gateway-macs', None) and conf['mclag-gateway-macs'].get('mclag-gateway-mac', None):
                gw_mac_data = conf['mclag-gateway-macs']['mclag-gateway-mac']
                if gw_mac_data[0].get('config', None) and gw_mac_data[0]['config'].get('gateway-mac', None):
                    config['gateway_mac'] = gw_mac_data[0]['config']['gateway-mac']
    return config