from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def determine_deref_aliases(derefAlias):
    mapping = {'never': ldap.DEREF_NEVER, 'search': ldap.DEREF_SEARCHING, 'base': ldap.DEREF_FINDING, 'always': ldap.DEREF_ALWAYS}
    result = None
    if derefAlias in mapping:
        result = mapping.get(derefAlias)
    return result