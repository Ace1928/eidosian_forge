from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
def reduce_on_network(value, network):
    """
    Reduces a list of addresses to only the addresses that match a given network.
    :param: value: The list of addresses to filter on.
    :param: network: The network to validate against.
    :return: The reduced list of addresses.
    """
    n = _address_normalizer(network)
    n_first = ipaddr(ipaddr(n, 'network') or ipaddr(n, 'address'), 'int')
    n_last = ipaddr(ipaddr(n, 'broadcast') or ipaddr(n, 'address'), 'int')
    r = []
    for address in value:
        a = _address_normalizer(address)
        a_first = ipaddr(ipaddr(a, 'network') or ipaddr(a, 'address'), 'int')
        a_last = ipaddr(ipaddr(a, 'broadcast') or ipaddr(a, 'address'), 'int')
        if _range_checker(a_first, n_first, n_last) and _range_checker(a_last, n_first, n_last):
            r.append(address)
    return r