from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_backup_resource_tags
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def tag_vault(module, client, tags, vault_arn, curr_tags=None, purge_tags=True):
    """
    Creates, updates, removes tags on a Backup Vault resource

    module : AnsibleAWSModule object
    client : boto3 client connection object
    tags : Dict of tags converted from ansible_dict to boto3 list of dicts
    vault_arn : The ARN of the Backup Vault to operate on
    curr_tags : Dict of the current tags on resource, if any
    purge_tags : true/false to determine if current tags will be retained or not
    """
    if tags is None:
        return False
    curr_tags = curr_tags or {}
    tags_to_add, tags_to_remove = compare_aws_tags(curr_tags, tags, purge_tags=purge_tags)
    if not tags_to_add and (not tags_to_remove):
        return False
    if module.check_mode:
        return True
    if tags_to_remove:
        try:
            client.untag_resource(ResourceArn=vault_arn, TagKeyList=tags_to_remove)
        except (BotoCoreError, ClientError) as err:
            module.fail_json_aws(err, msg='Failed to remove tags from the vault')
    if tags_to_add:
        try:
            client.tag_resource(ResourceArn=vault_arn, Tags=tags_to_add)
        except (BotoCoreError, ClientError) as err:
            module.fail_json_aws(err, msg='Failed to add tags to Vault')
    return True