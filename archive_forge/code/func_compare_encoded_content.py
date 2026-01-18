from __future__ import absolute_import, division, print_function
import base64
import json
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.sops.plugins.module_utils.io import write_file
from ansible_collections.community.sops.plugins.module_utils.sops import Sops, SopsError, get_sops_argument_spec
def compare_encoded_content(module, binary_data, content):
    if module.params['content_text'] is not None:
        return content == module.params['content_text'].encode('utf-8')
    if module.params['content_binary'] is not None:
        return content == binary_data
    if module.params['content_json'] is not None:
        try:
            return json.loads(content) == module.params['content_json']
        except Exception:
            return False
    if module.params['content_yaml'] is not None:
        try:
            return yaml.safe_load(content) == module.params['content_yaml']
        except Exception:
            return False
    module.fail_json(msg='Internal error: unknown content type')