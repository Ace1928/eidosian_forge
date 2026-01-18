from __future__ import absolute_import, division, print_function
import re
import time
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_mdt(configobj, name):
    cfg = configobj['vrf definition %s' % name]
    mdt_list = []
    for ip in ['ipv4', 'ipv6']:
        ret_dict = {}
        try:
            subcfg = cfg['address-family ' + ip]
            subcfg = '\n'.join(subcfg.children)
        except KeyError:
            subcfg = ''
            pass
        re1 = re.compile('^mdt +auto\\-discovery +(?P<option>\\S+)(\\s+(?P<inter_as>inter\\-as))?$')
        re2 = re.compile('^mdt +default +vxlan +(?P<mcast_group>\\S+)$')
        re3 = re.compile('^mdt +data +vxlan +(?P<mcast_group>.+)$')
        re4 = re.compile('^mdt +data +threshold +(?P<threshold_value>\\d+)$')
        re5 = re.compile('^mdt +overlay +(?P<use_bgp>use-bgp)(\\s+(?P<spt_only>spt-only))?$')
        for line in subcfg.splitlines():
            line = line.strip()
            m = re1.match(line)
            if m:
                group = m.groupdict()
                ret_dict.setdefault('auto_discovery', {}).setdefault(group['option'], {}).setdefault('enable', True)
                if group['inter_as']:
                    ret_dict.setdefault('auto_discovery', {}).setdefault(group['option'], {}).setdefault('inter_as', True)
                continue
            m = re2.match(line)
            if m:
                group = m.groupdict()
                ret_dict.setdefault('default', {}).setdefault('vxlan_mcast_group', group['mcast_group'])
                continue
            m = re3.match(line)
            if m:
                group = m.groupdict()
                ret_dict.setdefault('data_mcast', {}).setdefault('vxlan_mcast_group', group['mcast_group'])
                continue
            m = re4.match(line)
            if m:
                group = m.groupdict()
                ret_dict.setdefault('data_threshold', int(group['threshold_value']))
            m = re5.match(line)
            if m:
                group = m.groupdict()
                ret_dict.setdefault('overlay', {}).setdefault('use_bgp', {}).setdefault('enable', True)
                if group['spt_only']:
                    ret_dict.setdefault('overlay', {}).setdefault('use_bgp', {}).setdefault('spt_only', True)
        mdt_list.append({'afi': ip, 'mdt': ret_dict})
    return mdt_list