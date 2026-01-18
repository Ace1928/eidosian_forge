import json
from traceback import format_exc
from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def put_resource_policy(self, secret):
    if self.module.check_mode:
        self.module.exit_json(changed=True)
    try:
        json.loads(secret.secret_resource_policy_args.get('ResourcePolicy'))
    except (TypeError, ValueError) as e:
        self.module.fail_json(msg=f'Failed to parse resource policy as JSON: {str(e)}', exception=format_exc())
    try:
        response = self.client.put_resource_policy(**secret.secret_resource_policy_args)
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e, msg='Failed to update secret resource policy')
    return response