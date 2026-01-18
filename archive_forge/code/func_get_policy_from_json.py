import json
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def get_policy_from_json(self):
    if isinstance(self.policy_json, string_types):
        pdoc = json.loads(self.policy_json)
    else:
        pdoc = self.policy_json
    return pdoc