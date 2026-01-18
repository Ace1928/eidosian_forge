from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def openshift_ldap_get_attribute_for_entry(entry, attribute):
    attributes = [attribute]
    if isinstance(attribute, list):
        attributes = attribute
    for k in attributes:
        if k.lower() == 'dn':
            return entry[0]
        v = entry[1].get(k, None)
        if v:
            if isinstance(v, list):
                result = []
                for x in v:
                    if hasattr(x, 'decode'):
                        result.append(x.decode('utf-8'))
                    else:
                        result.append(x)
                return result
            else:
                return v.decode('utf-8') if hasattr(v, 'decode') else v
    return ''