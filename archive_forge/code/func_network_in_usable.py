from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
def network_in_usable(value, test):
    """
    Checks whether 'test' is a usable address or addresses in 'value'
    :param: value: The string representation of an address or network to test against.
    :param test: The string representation of an address or network to validate if it is within the range of 'value'.
    :return: bool
    """
    v = _address_normalizer(value)
    w = _address_normalizer(test)
    v_first = ipaddr(ipaddr(v, 'first_usable') or ipaddr(v, 'address'), 'int')
    v_last = ipaddr(ipaddr(v, 'last_usable') or ipaddr(v, 'address'), 'int')
    w_first = ipaddr(ipaddr(w, 'network') or ipaddr(w, 'address'), 'int')
    w_last = ipaddr(ipaddr(w, 'broadcast') or ipaddr(w, 'address'), 'int')
    if _range_checker(w_first, v_first, v_last) and _range_checker(w_last, v_first, v_last):
        return True
    else:
        return False