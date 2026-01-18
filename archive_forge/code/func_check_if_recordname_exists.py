from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def check_if_recordname_exists(self, obj_filter, ib_obj_ref, ib_obj_type, current_object, proposed_object):
    """ Send POST request if host record input name and retrieved ref name is same,
            but input IP and retrieved IP is different"""
    if 'name' in (obj_filter and ib_obj_ref[0]) and ib_obj_type == NIOS_HOST_RECORD:
        obj_host_name = obj_filter['name']
        ref_host_name = ib_obj_ref[0]['name']
        if 'ipv4addrs' in (current_object and proposed_object):
            current_ip_addr = current_object['ipv4addrs'][0]['ipv4addr']
            proposed_ip_addr = proposed_object['ipv4addrs'][0]['ipv4addr']
        elif 'ipv6addrs' in (current_object and proposed_object):
            current_ip_addr = current_object['ipv6addrs'][0]['ipv6addr']
            proposed_ip_addr = proposed_object['ipv6addrs'][0]['ipv6addr']
        if obj_host_name == ref_host_name and current_ip_addr != proposed_ip_addr:
            self.create_object(ib_obj_type, proposed_object)