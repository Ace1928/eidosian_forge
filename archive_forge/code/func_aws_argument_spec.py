import logging
import os
import re
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import boto3_at_least
from .botocore import boto3_conn
from .botocore import botocore_at_least
from .botocore import check_sdk_version_supported
from .botocore import gather_sdk_versions
from .botocore import get_aws_connection_info
from .botocore import get_aws_region
from .exceptions import AnsibleBotocoreError
from .retries import RetryingBotoClientWrapper
def aws_argument_spec():
    """
    Returns a dictionary containing the argument_spec common to all AWS modules.
    """
    region_spec = dict(region=dict(aliases=['aws_region', 'ec2_region'], deprecated_aliases=[dict(name='ec2_region', date='2024-12-01', collection_name='amazon.aws')], fallback=(env_fallback, ['AWS_REGION', 'AWS_DEFAULT_REGION', 'EC2_REGION'])))
    spec = _aws_common_argument_spec()
    spec.update(region_spec)
    return spec