from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_ssm_inventory(connection, filters):
    try:
        return connection.get_inventory(Filters=filters)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        raise SsmInventoryInfoFailure(exc=e, msg='get_ssm_inventory() failed.')