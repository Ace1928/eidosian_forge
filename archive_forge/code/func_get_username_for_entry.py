from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def get_username_for_entry(self, entry):
    username = openshift_ldap_get_attribute_for_entry(entry, self.ldap_interface.userNameAttributes)
    if not username:
        return (None, 'The user entry (%s) does not map to a OpenShift User name with the given mapping' % entry)
    return (username, None)