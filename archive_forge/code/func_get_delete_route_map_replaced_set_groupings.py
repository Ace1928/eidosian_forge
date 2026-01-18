from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_route_map_replaced_set_groupings(self, command, cmd_rmap_have, requests):
    """For the route map specified by the input "command", create requests
        to delete any existing route map "set" configuration groupings for which
        modified attribute requests are specified"""
    if not command.get('set'):
        return
    conf_map_name = command.get('map_name', None)
    conf_seq_num = command.get('sequence_num', None)
    req_seq_num = str(conf_seq_num)
    cmd_set_top = command['set']
    cfg_set_top = cmd_rmap_have.get('set')
    if not cfg_set_top:
        command.pop('set')
        return
    set_delete_req_base = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num) + 'actions/'
    bgp_set_delete_req_base = set_delete_req_base + 'openconfig-bgp-policy:bgp-actions/'
    cmd_set_keys = cmd_set_top.keys()
    cfg_set_keys = cfg_set_top.keys()
    metric_uri = ''
    if 'metric' in cfg_set_top:
        if cfg_set_top['metric'].get('rtt_action'):
            metric_uri = set_delete_req_base + 'metric-action/config'
        elif cfg_set_top['metric'].get('value'):
            metric_uri = [set_delete_req_base + 'metric-action/config', bgp_set_delete_req_base + 'config/set-med']
    set_top_level_keys = ['as_path_prepend', 'comm_list_delete', 'ip_next_hop', 'local_preference', 'metric', 'origin', 'weight']
    set_uri_attr = {'as_path_prepend': bgp_set_delete_req_base + 'set-as-path-prepend', 'comm_list_delete': bgp_set_delete_req_base + 'set-community-delete', 'community': bgp_set_delete_req_base + 'set-community', 'extcommunity': bgp_set_delete_req_base + 'set-ext-community', 'ip_next_hop': bgp_set_delete_req_base + 'config/set-next-hop', 'ipv6_next_hop': {'global_addr': bgp_set_delete_req_base + 'config/set-ipv6-next-hop-global', 'prefer_global': bgp_set_delete_req_base + 'config/set-ipv6-next-hop-prefer-global'}, 'local_preference': bgp_set_delete_req_base + 'config/set-local-pref', 'metric': metric_uri, 'origin': bgp_set_delete_req_base + 'config/set-route-origin', 'weight': bgp_set_delete_req_base + 'config/set-weight'}
    cfg_top_level_key_set = set(cfg_set_keys).intersection(set(set_top_level_keys))
    cmd_top_level_key_set = set(cmd_set_keys).intersection(set(set_top_level_keys))
    cmd_nested_level_key_set = set(cmd_set_keys).difference(set_top_level_keys)
    symmetric_diff_set = cmd_top_level_key_set.symmetric_difference(cfg_top_level_key_set)
    intersection_diff_set = cmd_top_level_key_set.intersection(cfg_top_level_key_set)
    cmd_delete_dict = {}
    if cmd_top_level_key_set and symmetric_diff_set or any((keyname for keyname in intersection_diff_set if cmd_set_top[keyname] != cfg_set_top[keyname])):
        self.delete_replaced_dict_config(cfg_key_set=cfg_top_level_key_set, cmd_key_set=cmd_top_level_key_set, cfg_parent_dict=cfg_set_top, uri_attr=set_uri_attr, uri_dict_key='cfg_dict_member_key', deletion_dict=cmd_delete_dict, requests=requests)
        cmd_set_nested = {}
        for nested_key in cmd_nested_level_key_set:
            if command['set'].get(nested_key) is not None:
                cmd_set_nested[nested_key] = command['set'][nested_key]
        command.pop('set')
        if cmd_delete_dict:
            command['set'] = cmd_delete_dict
        if cmd_set_nested:
            if not command.get('set'):
                command['set'] = {}
        command['set'].update(cmd_set_nested)
        if not command.get('set'):
            command['set'] = {}
        cmd_set_top = command['set']
        dict_delete_requests = []
        set_community_delete_attrs = []
        if 'community' not in cfg_set_top:
            if command['set'].get('community'):
                command['set'].pop('community')
                if command['set'] is None:
                    command.pop('set')
                return
        else:
            set_community_number_deletions = []
            if 'community_number' in cfg_set_top['community']:
                cfg_community_number_set = set(cfg_set_top['community']['community_number'])
                cmd_community_number_set = []
                if cmd_set_top.get('community') and 'community_number' in cmd_set_top['community']:
                    cmd_community_number_set = set(cmd_set_top['community']['community_number'])
                    command['set']['community'].pop('community_number')
                for cfg_community_number in cfg_community_number_set.difference(cmd_community_number_set):
                    set_community_delete_attrs.append(cfg_community_number)
                    set_community_number_deletions.append(cfg_community_number)
                if set_community_number_deletions:
                    if not cmd_set_top.get('community'):
                        command['set']['community'] = {}
                    command['set']['community']['community_number'] = set_community_number_deletions
            set_community_attributes_deletions = []
            if 'community_attributes' in cfg_set_top['community']:
                cfg_community_attributes_set = set(cfg_set_top['community']['community_attributes'])
                cmd_community_attributes_set = []
                if cmd_set_top.get('community') and 'community_attributes' in cmd_set_top['community']:
                    cmd_community_attributes_set = set(cmd_set_top['community']['community_attributes'])
                    command['set']['community'].pop('community_attributes')
                for cfg_community_attribute in cfg_community_attributes_set.difference(cmd_community_attributes_set):
                    set_community_delete_attrs.append(self.set_community_rest_names[cfg_community_attribute])
                    set_community_attributes_deletions.append(cfg_community_attribute)
                if set_community_attributes_deletions:
                    if not cmd_set_top.get('community'):
                        command['set']['community'] = {}
                    command['set']['community']['community_attributes'] = set_community_attributes_deletions
            if command['set'].get('community') is not None and (not command['set']['community']):
                command['set'].pop('community')
            if set_community_delete_attrs:
                bgp_set_delete_community_uri = bgp_set_delete_req_base + 'set-community'
                bgp_set_delete_comm_payload = {'openconfig-bgp-policy:set-community': {}}
                bgp_set_delete_comm_payload_contents = bgp_set_delete_comm_payload['openconfig-bgp-policy:set-community']
                bgp_set_delete_comm_payload_contents['config'] = {'method': 'INLINE', 'options': 'REMOVE'}
                bgp_set_delete_comm_payload_contents['inline'] = {'config': {'communities': set_community_delete_attrs}}
                request = {'path': bgp_set_delete_community_uri, 'method': PATCH, 'data': bgp_set_delete_comm_payload}
                dict_delete_requests.append(request)
        set_extcommunity_delete_attrs = []
        if 'extcommunity' not in cfg_set_top:
            if command['set'].get('extcommunity'):
                command['set'].pop('extcommunity')
                if command['set'] is None:
                    command.pop('set')
                return
        else:
            for extcomm_type in self.set_extcomm_rest_names:
                set_extcommunity_delete_attrs_type = []
                if extcomm_type in cfg_set_top['extcommunity']:
                    cfg_extcommunity_list_set = set(cfg_set_top['extcommunity'][extcomm_type])
                    cmd_extcommunity_list_set = []
                    if cmd_set_top.get('extcommunity') and extcomm_type in cmd_set_top['extcommunity']:
                        cmd_extcommunity_list_set = set(cmd_set_top['extcommunity'][extcomm_type])
                        command['set']['extcommunity'].pop(extcomm_type)
                    for extcomm_number in cfg_extcommunity_list_set.difference(cmd_extcommunity_list_set):
                        set_extcommunity_delete_attrs.append(self.set_extcomm_rest_names[extcomm_type] + extcomm_number)
                        set_extcommunity_delete_attrs_type.append(extcomm_number)
                    if set_extcommunity_delete_attrs_type:
                        if not cmd_set_top.get('extcommunity'):
                            command['set']['extcommunity'] = {}
                        command['set']['extcommunity'][extcomm_type] = set_extcommunity_delete_attrs_type
            if command['set'].get('extcommunity') is not None and (not command['set']['extcommunity']):
                command['set'].pop('extcommunity')
            if set_extcommunity_delete_attrs:
                bgp_set_delete_extcomm_uri = bgp_set_delete_req_base + 'set-ext-community'
                bgp_set_delete_extcomm_payload = {'openconfig-bgp-policy:set-ext-community': {}}
                bgp_set_delete_comm_payload_contents = bgp_set_delete_extcomm_payload['openconfig-bgp-policy:set-ext-community']
                bgp_set_delete_comm_payload_contents['config'] = {'method': 'INLINE', 'options': 'REMOVE'}
                bgp_set_delete_comm_payload_contents['inline'] = {'config': {'communities': set_extcommunity_delete_attrs}}
                request = {'path': bgp_set_delete_extcomm_uri, 'method': PATCH, 'data': bgp_set_delete_extcomm_payload}
                dict_delete_requests.append(request)
        ipv6_next_hop_deleted_members = {}
        if 'ipv6_next_hop' not in cfg_set_top:
            if command['set'].get('ipv6_next_hop'):
                command['set'].pop('ipv6_next_hop')
                if command['set'] is None:
                    command.pop('set')
                return
        else:
            cfg_ipv6_next_hop_key_set = set(cfg_set_top['ipv6_next_hop'].keys())
            cmd_ipv6_next_hop_key_set = []
            if cmd_set_top.get('ipv6_next_hop'):
                cmd_ipv6_next_hop_key_set = set(cfg_set_top['ipv6_next_hop'].keys())
                command['set'].pop('ipv6_next_hop')
            set_uri = set_uri_attr['ipv6_next_hop']
            for ipv6_next_hop_key in cfg_ipv6_next_hop_key_set.difference(cmd_ipv6_next_hop_key_set):
                ipv6_next_hop_deleted_members[ipv6_next_hop_key] = cfg_set_top['ipv6_next_hop'][ipv6_next_hop_key]
                request = {'path': set_uri[ipv6_next_hop_key], 'method': DELETE}
                dict_delete_requests.append(request)
            if ipv6_next_hop_deleted_members:
                if not cmd_set_top.get('ipv6_next_hop'):
                    command['set']['ipv6_next_hop'] = {}
                command['set']['ipv6_next_hop'] = ipv6_next_hop_deleted_members
        if dict_delete_requests:
            requests.extend(dict_delete_requests)
        return
    dict_delete_requests = []
    set_community_delete_attrs = []
    if 'community' in cmd_set_top:
        if 'community' not in cfg_set_top:
            command['set'].pop('community')
            if command['set'] is None:
                command.pop('set')
                return
        else:
            if 'community_number' in cmd_set_top['community']:
                set_community_number_deletions = []
                if 'community_number' in cfg_set_top['community']:
                    symmetric_diff_set = set(cmd_set_top['community']['community_number']).symmetric_difference(set(cfg_set_top['community']['community_number']))
                    if symmetric_diff_set:
                        for community_number in cfg_set_top['community']['community_number']:
                            if community_number not in cmd_set_top['community']['community_number']:
                                set_community_delete_attrs.append(community_number)
                                set_community_number_deletions.append(community_number)
                command['set']['community'].pop('community_number')
                if set_community_delete_attrs:
                    command['set']['community']['community_number'] = set_community_number_deletions
            if 'community_attributes' in cmd_set_top['community']:
                set_community_named_attr_deletions = []
                if 'community_attributes' in cfg_set_top['community']:
                    symmetric_diff_set = set(cmd_set_top['community']['community_attributes']).symmetric_difference(set(cfg_set_top['community']['community_attributes']))
                    if symmetric_diff_set:
                        cfg_set_top_comm_attr = cfg_set_top['community']['community_attributes']
                        for community_attr in cfg_set_top_comm_attr:
                            if community_attr not in cmd_set_top['community']['community_attributes']:
                                set_community_delete_attrs.append(self.set_community_rest_names[community_attr])
                                set_community_named_attr_deletions.append(community_attr)
                command['set']['community'].pop('community_attributes')
                if set_community_named_attr_deletions:
                    command['set']['community']['community_attributes'] = set_community_named_attr_deletions
            if command['set']['community'] is None:
                command['set'].pop('community')
            if set_community_delete_attrs:
                bgp_set_delete_community_uri = bgp_set_delete_req_base + 'set-community'
                bgp_set_delete_comm_payload = {'openconfig-bgp-policy:set-community': {}}
                bgp_set_delete_comm_payload_contents = bgp_set_delete_comm_payload['openconfig-bgp-policy:set-community']
                bgp_set_delete_comm_payload_contents['config'] = {'method': 'INLINE', 'options': 'REMOVE'}
                bgp_set_delete_comm_payload_contents['inline'] = {'config': {'communities': set_community_delete_attrs}}
                request = {'path': bgp_set_delete_community_uri, 'method': PATCH, 'data': bgp_set_delete_comm_payload}
                dict_delete_requests.append(request)
    set_extcommunity_delete_attrs = []
    if 'extcommunity' in cmd_set_top:
        if 'extcommunity' not in cfg_set_top:
            command['set'].pop('extcommunity')
        else:
            for extcomm_type in self.set_extcomm_rest_names:
                set_extcommunity_delete_attrs_type = []
                if cmd_set_top['extcommunity'].get(extcomm_type):
                    if extcomm_type in cfg_set_top['extcommunity']:
                        symmetric_diff_set = set(cmd_set_top['extcommunity'][extcomm_type]).symmetric_difference(set(cfg_set_top['extcommunity'][extcomm_type]))
                        if symmetric_diff_set:
                            for extcomm_number in cfg_set_top['extcommunity'][extcomm_type]:
                                if extcomm_number not in cmd_set_top['extcommunity'][extcomm_type]:
                                    set_extcommunity_delete_attrs.append(self.set_extcomm_rest_names[extcomm_type] + extcomm_number)
                                    set_extcommunity_delete_attrs_type.append(extcomm_number)
                    command['set']['extcommunity'].pop(extcomm_type)
                    if set_extcommunity_delete_attrs_type:
                        command['set']['extcommunity'][extcomm_type] = set_extcommunity_delete_attrs_type
            if command['set']['extcommunity'] is None:
                command['set'].pop('extcommunity')
            if set_extcommunity_delete_attrs:
                bgp_set_delete_extcomm_uri = bgp_set_delete_req_base + 'set-ext-community'
                bgp_set_delete_extcomm_payload = {'openconfig-bgp-policy:set-ext-community': {}}
                bgp_set_delete_comm_payload_contents = bgp_set_delete_extcomm_payload['openconfig-bgp-policy:set-ext-community']
                bgp_set_delete_comm_payload_contents['config'] = {'method': 'INLINE', 'options': 'REMOVE'}
                bgp_set_delete_comm_payload_contents['inline'] = {'config': {'communities': set_extcommunity_delete_attrs}}
                request = {'path': bgp_set_delete_extcomm_uri, 'method': PATCH, 'data': bgp_set_delete_extcomm_payload}
                dict_delete_requests.append(request)
    if 'ipv6_next_hop' in cmd_set_top:
        ipv6_next_hop_deleted_members = {}
        if 'ipv6_next_hop' in cfg_set_top:
            symmetric_diff_set = set(cmd_set_top['ipv6_next_hop'].keys()).symmetric_difference(set(cfg_set_top['ipv6_next_hop'].keys()))
            intersection_diff_set = set(cmd_set_top['ipv6_next_hop'].keys()).intersection(set(cfg_set_top['ipv6_next_hop'].keys()))
            if symmetric_diff_set or any((keyname for keyname in intersection_diff_set if cmd_set_top['ipv6_next_hop'][keyname] != cfg_set_top['ipv6_next_hop'][keyname])):
                set_uri = set_uri_attr['ipv6_next_hop']
                for member_key in set_uri:
                    if cfg_set_top['ipv6_next_hop'].get(member_key) is not None and cmd_set_top['ipv6_next_hop'].get(member_key) is None:
                        ipv6_next_hop_deleted_members[member_key] = cfg_set_top['ipv6_next_hop'][member_key]
                        request = {'path': set_uri[member_key], 'method': DELETE}
                        dict_delete_requests.append(request)
        command['set'].pop('ipv6_next_hop')
        if ipv6_next_hop_deleted_members:
            command['set']['ipv6_next_hop'] = ipv6_next_hop_deleted_members
    if dict_delete_requests:
        requests.extend(dict_delete_requests)