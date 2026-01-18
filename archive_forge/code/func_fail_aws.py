from ansible.errors import AnsibleError
from ansible.module_utils.basic import to_native
from ansible.utils.display import Display
from ansible_collections.amazon.aws.plugins.module_utils.botocore import check_sdk_version_supported
from ansible_collections.amazon.aws.plugins.module_utils.retries import RetryingBotoClientWrapper
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import boto3_conn
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import get_aws_connection_info
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import get_aws_region
def fail_aws(self, message, exception=None):
    if not exception:
        self._do_fail(to_native(message))
    self._do_fail(f'{message}: {to_native(exception)}')