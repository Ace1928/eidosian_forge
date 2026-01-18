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
def gather_sdk_versions():
    """Gather AWS SDK (boto3 and botocore) dependency versions

    Returns {'boto3_version': str, 'botocore_version': str}
    Returns {} if either module is not installed
    """
    if not HAS_BOTO3:
        return {}
    return dict(boto3_version=boto3.__version__, botocore_version=botocore.__version__)