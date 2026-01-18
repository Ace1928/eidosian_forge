from __future__ import absolute_import, division, print_function
import binascii
from collections import defaultdict
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def lookup_adminstatus(int_adminstatus):
    adminstatus_options = {1: 'up', 2: 'down', 3: 'testing'}
    if int_adminstatus in adminstatus_options:
        return adminstatus_options[int_adminstatus]
    return ''