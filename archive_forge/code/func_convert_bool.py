from __future__ import absolute_import, division, print_function
import os
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils._stormssh import ConfigParser, HAS_PARAMIKO, PARAMIKO_IMPORT_ERROR
from ansible_collections.community.general.plugins.module_utils.ssh import determine_config_file
def convert_bool(value):
    if value is True:
        return 'yes'
    if value is False:
        return 'no'
    return None