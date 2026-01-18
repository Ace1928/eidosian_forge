import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def setup_module_object():
    argument_spec = dict(state=dict(default='present', choices=['present', 'absent']), function_name=dict(required=True, aliases=['lambda_function_arn', 'function_arn']), statement_id=dict(required=True, aliases=['sid']), alias=dict(), version=dict(type='int'), action=dict(required=True), principal=dict(required=True), source_arn=dict(), source_account=dict(), event_source_token=dict(no_log=False))
    return AnsibleAWSModule(argument_spec=argument_spec, supports_check_mode=True, mutually_exclusive=[['alias', 'version'], ['event_source_token', 'source_arn'], ['event_source_token', 'source_account']])