from __future__ import (absolute_import, division, print_function)
import re
from random import Random, SystemRandom
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
def random_mac(value, seed=None):
    """ takes string prefix, and return it completed with random bytes
        to get a complete 6 bytes MAC address """
    if not isinstance(value, string_types):
        raise AnsibleFilterError('Invalid value type (%s) for random_mac (%s)' % (type(value), value))
    value = value.lower()
    mac_items = value.split(':')
    if len(mac_items) > 5:
        raise AnsibleFilterError('Invalid value (%s) for random_mac: 5 colon(:) separated items max' % value)
    err = ''
    for mac in mac_items:
        if not mac:
            err += ',empty item'
            continue
        if not re.match('[a-f0-9]{2}', mac):
            err += ',%s not hexa byte' % mac
    err = err.strip(',')
    if err:
        raise AnsibleFilterError('Invalid value (%s) for random_mac: %s' % (value, err))
    if seed is None:
        r = SystemRandom()
    else:
        r = Random(seed)
    v = r.randint(68719476736, 1099511627775)
    remain = 2 * (6 - len(mac_items))
    rnd = ('%x' % v)[:remain]
    return value + re.sub('(..)', ':\\1', rnd)