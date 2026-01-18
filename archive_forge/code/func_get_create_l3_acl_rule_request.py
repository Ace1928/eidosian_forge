from __future__ import absolute_import, division, print_function
from ast import literal_eval
from ansible.module_utils._text import to_text
from ansible.module_utils.common.validation import check_required_arguments
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_create_l3_acl_rule_request(self, acl_type, acl_name, seq_num, rule):
    """Get request to create a rule with given sequence number
        and configuration in the specified L3 ACL
        """
    url = self.l3_acl_rule_path.format(acl_name=acl_name, acl_type=acl_type_to_payload_map[acl_type])
    payload = {'openconfig-acl:acl-entry': [{'sequence-id': seq_num, 'config': {'sequence-id': seq_num}, acl_type: {'config': {}}, 'transport': {'config': {}}, 'actions': {'config': {'forwarding-action': action_value_to_payload_map[rule['action']]}}}]}
    rule_l3_config = payload['openconfig-acl:acl-entry'][0][acl_type]['config']
    rule_l4_config = payload['openconfig-acl:acl-entry'][0]['transport']['config']
    if rule['protocol'].get('number') is not None:
        protocol = rule['protocol']['number']
        rule_l3_config['protocol'] = protocol_number_to_payload_map.get(protocol, protocol)
    else:
        protocol = rule['protocol']['name']
        if protocol not in ('ip', 'ipv6'):
            rule_l3_config['protocol'] = protocol_name_to_payload_map[protocol]
    if rule['source'].get('host'):
        rule_l3_config['source-address'] = rule['source']['host'] + acl_type_to_host_mask_map[acl_type]
    elif rule['source'].get('prefix'):
        rule_l3_config['source-address'] = rule['source']['prefix']
    src_port_number = self._convert_port_dict_to_payload_format(rule['source'].get('port_number'))
    if src_port_number:
        rule_l4_config['source-port'] = src_port_number
    if rule['destination'].get('host'):
        rule_l3_config['destination-address'] = rule['destination']['host'] + acl_type_to_host_mask_map[acl_type]
    elif rule['destination'].get('prefix'):
        rule_l3_config['destination-address'] = rule['destination']['prefix']
    dest_port_number = self._convert_port_dict_to_payload_format(rule['destination'].get('port_number'))
    if dest_port_number:
        rule_l4_config['destination-port'] = dest_port_number
    if rule.get('protocol_options'):
        if protocol in ('icmp', 'icmpv6') and rule['protocol_options'].get(protocol):
            if rule['protocol_options'][protocol].get('type') is not None:
                rule_l4_config['icmp-type'] = rule['protocol_options'][protocol]['type']
            if rule['protocol_options'][protocol].get('code') is not None:
                rule_l4_config['icmp-code'] = rule['protocol_options'][protocol]['code']
        elif rule['protocol_options'].get('tcp'):
            if rule['protocol_options']['tcp'].get('established'):
                rule_l4_config['tcp-session-established'] = True
            else:
                tcp_flag_list = []
                for tcp_flag in rule['protocol_options']['tcp'].keys():
                    if rule['protocol_options']['tcp'][tcp_flag]:
                        tcp_flag_list.append('tcp_{0}'.format(tcp_flag).upper())
                if tcp_flag_list:
                    rule_l4_config['tcp-flags'] = tcp_flag_list
    if rule.get('vlan_id') is not None:
        payload['openconfig-acl:acl-entry'][0]['l2'] = {'config': {'vlanid': rule['vlan_id']}}
    if rule.get('dscp'):
        if rule['dscp'].get('value') is not None:
            rule_l3_config['dscp'] = rule['dscp']['value']
        else:
            dscp_opt = next(iter(rule['dscp']))
            if rule['dscp'][dscp_opt]:
                rule_l3_config['dscp'] = dscp_name_to_value_map[dscp_opt]
    if rule.get('remark'):
        payload['openconfig-acl:acl-entry'][0]['config']['description'] = rule['remark']
    return {'path': url, 'method': POST, 'data': payload}