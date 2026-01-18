from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.logging_global.logging_global import (
def parse_host_node(self, conf):
    hosts_list = []
    hosts = []
    if isinstance(conf, dict):
        hosts.append(conf)
    else:
        hosts = conf
    for host in hosts:
        host_dict = {}
        host_dict['name'] = host.get('name')
        if 'allow-duplicates' in host:
            host_dict['allow_duplicates'] = True
        if 'contents' in host.keys():
            contents = host.get('contents')
            if isinstance(contents, list):
                for content in contents:
                    host_dict = self.parse_console_node(content, host_dict)
            else:
                host_dict = self.parse_console_node(contents, host_dict)
        if 'exclude-hostname' in host.keys():
            host_dict['exclude_hostname'] = True
        if 'facility-override' in host.keys():
            host_dict['facility_override'] = host.get('facility-override')
        if 'log-prefix' in host.keys():
            host_dict['log_prefix'] = host.get('log-prefix')
        if 'match' in host.keys():
            host_dict['match'] = host.get('match')
        if 'match-strings' in host.keys():
            match_strings = host.get('match-strings')
            match_strings_list = []
            if isinstance(match_strings, list):
                for item in match_strings:
                    match_strings_list.append(item)
            else:
                match_strings_list.append(match_strings)
            host_dict['match_strings'] = match_strings_list
        if 'port' in host.keys():
            host_dict['port'] = host.get('port')
        if 'routing-instance' in host.keys():
            host_dict['routing_instance'] = host.get('routing-instance')
        if 'source-address' in host.keys():
            host_dict['source_address'] = host.get('source-address')
        if 'structured-data' in host.keys():
            structured_data_dict = {}
            if host.get('structured-data'):
                structured_data_dict['brief'] = True
            else:
                structured_data_dict['set'] = True
            host_dict['structured_data'] = structured_data_dict
        hosts_list.append(host_dict)
    return hosts_list