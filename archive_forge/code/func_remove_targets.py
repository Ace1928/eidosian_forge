import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def remove_targets(self, target_ids):
    """Removes the provided targets from the rule in AWS"""
    if not target_ids:
        return
    request = {'Rule': self.name, 'Ids': target_ids}
    try:
        response = self.client.remove_targets(**request)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg=f'Could not remove rule targets from rule {self.name}')
    self.changed = True
    return response