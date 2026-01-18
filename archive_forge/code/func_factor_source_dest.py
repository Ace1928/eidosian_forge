from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.acls.acls import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def factor_source_dest(ace, typ):
    temp = ace.get(typ, {})
    if temp.get('address'):
        _temp_addr = temp.get('address', '')
        ace[typ]['address'] = _temp_addr.split(' ')[0]
        ace[typ]['wildcard_bits'] = _temp_addr.split(' ')[1]
    if temp.get('ipv6_address'):
        _temp_addr = temp.get('ipv6_address', '')
        if len(_temp_addr.split(' ')) == 2:
            ipv6_add = ace[typ].pop('ipv6_address')
            ace[typ]['address'] = ipv6_add.split(' ')[0]
            ace[typ]['wildcard_bits'] = ipv6_add.split(' ')[1]
        else:
            ace[typ]['address'] = ace[typ].pop('ipv6_address')