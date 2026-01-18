from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def check_n_return_valid_ipv6_addr(module, input_list, filtered_ipv6_list):
    try:
        for each in input_list:
            if '::' in each:
                if '/' in each:
                    each = each.split('/')[0]
                if socket.inet_pton(socket.AF_INET6, each):
                    filtered_ipv6_list.append(each)
        return filtered_ipv6_list
    except socket.error:
        module.fail_json(msg='Incorrect IPV6 address!')