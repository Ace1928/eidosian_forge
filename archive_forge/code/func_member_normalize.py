from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def member_normalize(member_spec):
    """ Transforms the member module arguments into a valid WAPI struct
    This function will transform the arguments into a structure that
    is a valid WAPI structure in the format of:
        {
            key: <value>,
        }
    It will remove any arguments that are set to None since WAPI will error on
    that condition.
    The remainder of the value validation is performed by WAPI
    Some parameters in ib_spec are passed as a list in order to pass the validation for elements.
    In this function, they are converted to dictionary.
    """
    member_elements = ['vip_setting', 'ipv6_setting', 'lan2_port_setting', 'mgmt_port_setting', 'pre_provisioning', 'network_setting', 'v6_network_setting', 'ha_port_setting', 'lan_port_setting', 'lan2_physical_setting', 'lan_ha_port_setting', 'mgmt_network_setting', 'v6_mgmt_network_setting']
    for key in list(member_spec.keys()):
        if key in member_elements and member_spec[key] is not None:
            member_spec[key] = member_spec[key][0]
        if isinstance(member_spec[key], dict):
            member_spec[key] = member_normalize(member_spec[key])
        elif isinstance(member_spec[key], list):
            for x in member_spec[key]:
                if isinstance(x, dict):
                    x = member_normalize(x)
        elif member_spec[key] is None:
            del member_spec[key]
    return member_spec