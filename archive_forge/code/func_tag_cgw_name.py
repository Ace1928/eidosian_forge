from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def tag_cgw_name(self, gw_id, name):
    response = self.ec2.create_tags(DryRun=False, Resources=[gw_id], Tags=[{'Key': 'Name', 'Value': name}])
    return response