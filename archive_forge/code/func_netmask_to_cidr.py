from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def netmask_to_cidr(netmask):
    return str(sum([bin(int(x)).count('1') for x in netmask.split('.')]))