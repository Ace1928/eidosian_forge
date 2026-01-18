from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def ip_tmplt(config_data):
    cmd = 'ipv6 address {ip}'
    if config_data.get('ipv6'):
        config = config_data.get('ipv6')
        cmd = cmd.format(ip=config['address'])
    if config.get('segment_routing'):
        cmd += ' segment-routing'
        if config.get('segment_routing').get('default'):
            cmd += ' default'
        if config.get('segment_routing').get('ipv6_sr'):
            cmd += ' ipv6-sr'
    if config.get('secondary'):
        cmd += ' secondary'
    if config.get('link_local'):
        cmd += ' link-local'
    if config.get('anycast'):
        cmd += ' anycast'
    if config.get('cga'):
        cmd += ' cga'
    if config.get('eui'):
        cmd += ' eui'
    return cmd