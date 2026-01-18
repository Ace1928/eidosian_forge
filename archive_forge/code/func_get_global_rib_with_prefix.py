import logging
import os
import netaddr
from . import docker_base as base
def get_global_rib_with_prefix(self, prefix, rf):
    rib = []
    lines = [line.strip() for line in self.vtysh('show bgp {0} unicast {1}'.format(rf, prefix), config=False).split('\n')]
    if lines[0] == '% Network not in table':
        return rib
    lines = lines[2:]
    if lines[0].startswith('Not advertised'):
        lines.pop(0)
    elif lines[0].startswith('Advertised to non peer-group peers:'):
        lines = lines[2:]
    else:
        raise Exception('unknown output format {0}'.format(lines))
    if lines[0] == 'Local':
        aspath = []
    else:
        aspath = [int(asn) for asn in lines[0].split()]
    nexthop = lines[1].split()[0].strip()
    info = [s.strip(',') for s in lines[2].split()]
    attrs = []
    if 'metric' in info:
        med = info[info.index('metric') + 1]
        attrs.append({'type': base.BGP_ATTR_TYPE_MULTI_EXIT_DISC, 'metric': int(med)})
    if 'localpref' in info:
        localpref = info[info.index('localpref') + 1]
        attrs.append({'type': base.BGP_ATTR_TYPE_LOCAL_PREF, 'value': int(localpref)})
    rib.append({'prefix': prefix, 'nexthop': nexthop, 'aspath': aspath, 'attrs': attrs})
    return rib