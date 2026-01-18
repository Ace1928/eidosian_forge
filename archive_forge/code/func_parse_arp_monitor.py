from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.lag_interfaces.lag_interfaces import (
def parse_arp_monitor(self, conf):
    arp_monitor = None
    if conf:
        arp_monitor = {}
        target_list = []
        interval = search('^.*arp-monitor interval (.+)', conf, M)
        targets = findall("^.*arp-monitor target '(.+)'", conf, M)
        if targets:
            for target in targets:
                target_list.append(target)
            arp_monitor['target'] = target_list
        if interval:
            value = interval.group(1).strip("'")
            arp_monitor['interval'] = int(value)
    return arp_monitor