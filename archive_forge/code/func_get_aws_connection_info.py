import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def get_aws_connection_info(module, boto3=None):
    try:
        return _aws_connection_info(module.params)
    except AnsibleBotocoreError as e:
        if e.exception:
            module.fail_json(msg=e.message, exception=e.exception)
        else:
            module.fail_json(msg=e.message)