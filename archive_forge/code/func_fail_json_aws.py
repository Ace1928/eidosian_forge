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
def fail_json_aws(self, exception, msg=None, **kwargs):
    """call fail_json with processed exception

        function for converting exceptions thrown by AWS SDK modules,
        botocore, boto3 and boto, into nice error messages.
        """
    last_traceback = traceback.format_exc()
    try:
        except_msg = to_native(exception.message)
    except AttributeError:
        except_msg = to_native(exception)
    if msg is not None:
        message = f'{msg}: {except_msg}'
    else:
        message = except_msg
    try:
        response = exception.response
    except AttributeError:
        response = None
    failure = dict(msg=message, exception=last_traceback, **self._gather_versions())
    failure.update(kwargs)
    if response is not None:
        failure.update(**camel_dict_to_snake_dict(response))
    self.fail_json(**failure)