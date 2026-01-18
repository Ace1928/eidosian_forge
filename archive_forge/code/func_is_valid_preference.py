from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def is_valid_preference(pref):
    """check if the preference is valid"""
    if int(pref) > 0 and int(pref) < 256:
        return True
    return False