import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def put_targets(self, targets):
    """Creates or updates the provided targets on the rule in AWS"""
    if not targets:
        return
    request = {'Rule': self.name, 'Targets': self._targets_request(targets)}
    try:
        response = self.client.put_targets(**request)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg=f'Could not create/update rule targets for rule {self.name}')
    self.changed = True
    return response