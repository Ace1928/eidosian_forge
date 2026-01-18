from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_local_disks(self, gateway):
    try:
        gateway['local_disks'] = [camel_dict_to_snake_dict(disk) for disk in self.client.list_local_disks(GatewayARN=gateway['gateway_arn'])['Disks']]
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e, msg="Couldn't list storage gateway local disks")