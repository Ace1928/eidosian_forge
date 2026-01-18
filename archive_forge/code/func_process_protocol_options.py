from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.acls.acls import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def process_protocol_options(each):
    for each_ace in each.get('aces'):
        if each.get('acl_type') == 'standard':
            if len(each_ace.get('source', {})) == 1 and each_ace.get('source', {}).get('address'):
                each_ace['source']['host'] = each_ace['source'].pop('address')
            if each_ace.get('source', {}).get('address'):
                addr = each_ace.get('source', {}).get('address')
                if addr[-1] == ',':
                    each_ace['source']['address'] = addr[:-1]
        else:
            if each_ace.get('source', {}):
                factor_source_dest(each_ace, 'source')
            if each_ace.get('destination', {}):
                factor_source_dest(each_ace, 'destination')
        if each_ace.get('icmp_igmp_tcp_protocol'):
            each_ace['protocol_options'] = {each_ace['protocol']: {each_ace.pop('icmp_igmp_tcp_protocol').replace('-', '_'): True}}
        if each_ace.get('protocol_number'):
            each_ace['protocol_options'] = {'protocol_number': each_ace.pop('protocol_number')}