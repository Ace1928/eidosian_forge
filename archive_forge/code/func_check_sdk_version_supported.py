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
def check_sdk_version_supported(botocore_version=None, boto3_version=None, warn=None):
    """Checks to see if the available boto3 / botocore versions are supported
    args:
        botocore_version: (str) overrides the minimum version of botocore supported by the collection
        boto3_version: (str) overrides the minimum version of boto3 supported by the collection
        warn: (Callable) invoked with a string message if boto3/botocore are less than the
            supported versions
    raises:
        AnsibleBotocoreError - If botocore/boto3 is missing
    returns
        False if boto3 or botocore is less than the minimum supported versions
        True if boto3 and botocore are greater than or equal the the minimum supported versions
    """
    botocore_version = botocore_version or MINIMUM_BOTOCORE_VERSION
    boto3_version = boto3_version or MINIMUM_BOTO3_VERSION
    if not HAS_BOTO3:
        raise AnsibleBotocoreError(message=missing_required_lib('botocore and boto3'))
    supported = True
    if not HAS_PACKAGING:
        if warn:
            warn('packaging.version Python module not installed, unable to check AWS SDK versions')
        return True
    if not botocore_at_least(botocore_version):
        supported = False
        if warn:
            warn(f'botocore < {MINIMUM_BOTOCORE_VERSION} is not supported or tested.  Some features may not work.')
    if not boto3_at_least(boto3_version):
        supported = False
        if warn:
            warn(f'boto3 < {MINIMUM_BOTO3_VERSION} is not supported or tested.  Some features may not work.')
    return supported