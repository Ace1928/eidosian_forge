from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_backup_resource_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_backup_vault_detail(connection, module):
    output = []
    result = {}
    backup_vault_name_list = module.params.get('backup_vault_names')
    if not backup_vault_name_list:
        backup_vault_name_list = get_backup_vaults(connection, module)
    for name in backup_vault_name_list:
        try:
            output.append(connection.describe_backup_vault(BackupVaultName=name, aws_retry=True))
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f'Failed to describe vault {name}')
    snaked_backup_vault = []
    for backup_vault in output:
        try:
            resource = backup_vault.get('BackupVaultArn', None)
            tag_dict = get_backup_resource_tags(module, connection, resource)
            backup_vault.update({'tags': tag_dict})
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.warn(f'Failed to get the backup vault tags - {e}')
        snaked_backup_vault.append(camel_dict_to_snake_dict(backup_vault))
    for v in snaked_backup_vault:
        if 'tags_list' in v:
            v['tags'] = boto3_tag_list_to_ansible_dict(v['tags_list'], 'key', 'value')
            del v['tags_list']
        if 'response_metadata' in v:
            del v['response_metadata']
    result['backup_vaults'] = snaked_backup_vault
    return result