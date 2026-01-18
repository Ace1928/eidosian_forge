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
def require_boto3_at_least(self, desired, **kwargs):
    """Check if the available boto3 version is greater than or equal to a desired version.

        calls fail_json() when the boto3 version is less than the desired
        version

        Usage:
            module.require_boto3_at_least("1.2.3", reason="to update tags")
            module.require_boto3_at_least("1.1.1")

        :param desired the minimum desired version
        :param reason why the version is required (optional)
        """
    if not self.boto3_at_least(desired):
        self._module.fail_json(msg=missing_required_lib(f'boto3>={desired}', **kwargs), **self._gather_versions())