from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def filter_dict_having_none_value(want, have):
    test_dict = dict()
    name = want.get('name')
    if name:
        test_dict['name'] = name
    diff_ip = False
    for k, v in iteritems(want):
        if isinstance(v, dict):
            for key, value in iteritems(v):
                test_key_dict = dict()
                if value is None:
                    if have.get(k):
                        dict_val = have.get(k).get(key)
                        test_key_dict.update({key: dict_val})
                elif k == 'ipv6' and value.lower() != have.get(k)[0].get(key).lower():
                    dict_val = have.get(k)[0].get(key)
                    test_key_dict.update({key: dict_val})
                if test_key_dict:
                    test_dict.update({k: test_key_dict})
        if isinstance(v, list):
            for key, value in iteritems(v[0]):
                test_key_dict = dict()
                if value is None:
                    if have.get(k) and key in have.get(k):
                        dict_val = have.get(k)[0].get(key)
                        test_key_dict.update({key: dict_val})
                elif have.get(k):
                    if k == 'ipv6' and value.lower() != have.get(k)[0].get(key).lower():
                        dict_val = have.get(k)[0].get(key)
                        test_key_dict.update({key: dict_val})
                if test_key_dict:
                    test_dict.update({k: test_key_dict})
            for each in v:
                if each.get('secondary'):
                    want_ip = each.get('address').split('/')
                    have_ip = have.get('ipv4')
                    if len(want_ip) > 1 and have_ip and have_ip[0].get('secondary'):
                        have_ip = have_ip[0]['address'].split(' ')[0]
                        if have_ip != want_ip[0]:
                            diff_ip = True
                    if each.get('secondary') and diff_ip is True:
                        test_key_dict.update({'secondary': True})
                    test_dict.update({'ipv4': test_key_dict})
        if v is None:
            val = have.get(k)
            test_dict.update({k: val})
    return test_dict