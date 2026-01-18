from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, replace_resource_dict
from ansible.module_utils._text import to_native
import json
import os
import base64
def self_link_from_file(module):
    key_name = key_name_from_file(module.params['path'], module)
    return 'https://iam.googleapis.com/v1/{key_name}'.format(key_name=key_name)