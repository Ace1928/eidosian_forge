from __future__ import (absolute_import, division, print_function)
import re
def base_dn():
    return config_registry()['ldap/base']