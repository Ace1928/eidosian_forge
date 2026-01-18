from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
def avi_obj_cmp(x, y, sensitive_fields=None):
    """
    compares whether x is fully contained in y. The comparision is different
    from a simple dictionary compare for following reasons
    1. Some fields could be references. The object in controller returns the
        full URL for those references. However, the ansible script would have
        it specified as /api/pool?name=blah. So, the reference fields need
        to match uuid, relative reference based on name and actual reference.

    2. Optional fields with defaults: In case there are optional fields with
        defaults then controller automatically fills it up. This would
        cause the comparison with Ansible object specification to always return
        changed.

    3. Optional fields without defaults: This is most tricky. The issue is
        how to specify deletion of such objects from ansible script. If the
        ansible playbook has object specified as Null then Avi controller will
        reject for non Message(dict) type fields. In addition, to deal with the
        defaults=null issue all the fields that are set with None are purged
        out before comparing with Avi controller's version

        So, the solution is to pass state: absent if any optional field needs
        to be deleted from the configuration. The script would return changed
        =true if it finds a key in the controller version and it is marked with
        state: absent in ansible playbook. Alternatively, it would return
        false if key is not present in the controller object. Before, doing
        put or post it would purge the fields that are marked state: absent.

    :param x: first string
    :param y: second string from controller's object
    :param sensitive_fields: sensitive fields to ignore for diff

    Returns:
        True if x is subset of y else False
    """
    if not sensitive_fields:
        sensitive_fields = set()
    if isinstance(x, str) or isinstance(x, str):
        return ref_n_str_cmp(x, y)
    if type(x) not in [list, dict]:
        return x == y
    if type(x) is list:
        if len(x) != len(y):
            log.debug('x has %d items y has %d', len(x), len(y))
            return False
        for i in zip(x, y):
            if not avi_obj_cmp(i[0], i[1], sensitive_fields=sensitive_fields):
                return False
    if type(x) is dict:
        x.pop('_last_modified', None)
        x.pop('tenant', None)
        y.pop('_last_modified', None)
        x.pop('api_version', None)
        y.pop('api_verison', None)
        d_xks = [k for k in x.keys() if k in sensitive_fields]
        if d_xks:
            return False
        d_x_absent_ks = []
        for k, v in x.items():
            if v is None:
                d_x_absent_ks.append(k)
                continue
            if isinstance(v, dict):
                if 'state' in v and v['state'] == 'absent':
                    if type(y) is dict and k not in y:
                        d_x_absent_ks.append(k)
                    else:
                        return False
                elif not v:
                    d_x_absent_ks.append(k)
            elif isinstance(v, list) and (not v):
                d_x_absent_ks.append(k)
            elif isinstance(v, str) or (k in y and isinstance(y[k], str)):
                if v == "{'state': 'absent'}" and k not in y:
                    d_x_absent_ks.append(k)
                elif not v and k not in y:
                    d_x_absent_ks.append(k)
        for k in d_x_absent_ks:
            x.pop(k)
        x_keys = set(x.keys())
        y_keys = set(y.keys())
        if not x_keys.issubset(y_keys):
            return False
        for k, v in x.items():
            if k not in y:
                return False
            if not avi_obj_cmp(v, y[k], sensitive_fields=sensitive_fields):
                return False
    return True