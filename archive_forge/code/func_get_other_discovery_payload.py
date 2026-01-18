from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def get_other_discovery_payload(module):
    trans_dict = {'discovery_job_name': 'DiscoveryConfigGroupName', 'trap_destination': 'TrapDestination', 'community_string': 'CommunityString', 'email_recipient': 'DiscoveryStatusEmailRecipient'}
    other_dict = {}
    for key, val in trans_dict.items():
        if module.params.get(key) is not None:
            other_dict[val] = module.params.get(key)
    return other_dict