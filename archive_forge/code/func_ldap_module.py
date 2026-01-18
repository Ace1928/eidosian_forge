from __future__ import (absolute_import, division, print_function)
import re
def ldap_module():
    import ldap as orig_ldap
    return orig_ldap