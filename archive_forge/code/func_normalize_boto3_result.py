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
def normalize_boto3_result(result):
    """
    Because Boto3 returns datetime objects where it knows things are supposed to
    be dates we need to mass-convert them over to strings which Ansible/Jinja
    handle better.  This also makes it easier to compare complex objects which
    include a mix of dates in string format (from parameters) and dates as
    datetime objects.  Boto3 is happy to be passed ISO8601 format strings.
    """
    return json.loads(json.dumps(result, default=_boto3_handler))