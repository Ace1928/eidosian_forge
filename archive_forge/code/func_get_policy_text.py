import json
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def get_policy_text(self):
    try:
        if self.policy_json is not None:
            return self.get_policy_from_json()
    except json.JSONDecodeError as e:
        raise PolicyError(f'Failed to decode the policy as valid JSON: {str(e)}')
    return None