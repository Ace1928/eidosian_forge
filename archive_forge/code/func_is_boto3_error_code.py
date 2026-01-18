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
def is_boto3_error_code(code, e=None):
    """Check if the botocore exception is raised by a specific error code.

    Returns ClientError if the error code matches, a dummy exception if it does not have an error code or does not match

    Example:
    try:
        ec2.describe_instances(InstanceIds=['potato'])
    except is_boto3_error_code('InvalidInstanceID.Malformed'):
        # handle the error for that code case
    except botocore.exceptions.ClientError as e:
        # handle the generic error case for all other codes
    """
    from botocore.exceptions import ClientError
    if e is None:
        import sys
        dummy, e, dummy = sys.exc_info()
    if not isinstance(code, list):
        code = [code]
    if isinstance(e, ClientError) and e.response['Error']['Code'] in code:
        return ClientError
    return type('NeverEverRaisedException', (Exception,), {})