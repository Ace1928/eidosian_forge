from ansible.module_utils.network.aos.aos import (check_aos_version, get_aos_session, find_collection_item,
from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.version import LooseVersion
from ansible.module_utils._text import to_native
def content_to_dict(module, content):
    """
    Convert 'content' into a Python Dict based on 'content_format'
    """
    content_dict = None
    try:
        content_dict = yaml.safe_load(content)
        if not isinstance(content_dict, dict):
            raise Exception()
        if not content_dict:
            raise Exception()
    except Exception:
        module.fail_json(msg="Unable to convert 'content' to a dict, please check if valid")
    module.params['content'] = content_dict
    return content_dict