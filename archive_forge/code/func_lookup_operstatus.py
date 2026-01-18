from __future__ import absolute_import, division, print_function
import binascii
from collections import defaultdict
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def lookup_operstatus(int_operstatus):
    operstatus_options = {1: 'up', 2: 'down', 3: 'testing', 4: 'unknown', 5: 'dormant', 6: 'notPresent', 7: 'lowerLayerDown'}
    if int_operstatus in operstatus_options:
        return operstatus_options[int_operstatus]
    return ''