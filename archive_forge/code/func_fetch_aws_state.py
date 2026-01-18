import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def fetch_aws_state(self):
    """Retrieves rule and target state from AWS"""
    aws_state = {'rule': {}, 'targets': [], 'changed': self.rule.changed}
    rule_description = self.rule.describe()
    if not rule_description:
        return aws_state
    del rule_description['response_metadata']
    aws_state['rule'] = rule_description
    aws_state['targets'].extend(self.rule.list_targets())
    return aws_state