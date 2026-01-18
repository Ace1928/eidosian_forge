from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_boto3_client_method_parameters
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import arg_spec_to_rds_params
from ansible_collections.amazon.aws.plugins.module_utils.rds import call_method
from ansible_collections.amazon.aws.plugins.module_utils.rds import compare_iam_roles
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_final_identifier
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_rds_method_attribute
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import update_iam_roles
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def ensure_iam_roles(client, module, instance_id):
    """
    Ensure specified IAM roles are associated with DB instance

        Parameters:
            client: RDS client
            module: AWSModule
            instance_id: DB's instance ID

        Returns:
            changed (bool): True if changes were successfully made to DB instance's IAM roles; False if not
    """
    instance = camel_dict_to_snake_dict(get_instance(client, module, instance_id), ignore_list=['Tags', 'ProcessorFeatures'])
    engine = instance.get('engine')
    if engine not in valid_engines_iam_roles:
        module.fail_json(msg=f'DB engine {engine} is not valid for adding IAM roles. Valid engines are {valid_engines_iam_roles}')
    changed = False
    purge_iam_roles = module.params.get('purge_iam_roles')
    target_roles = module.params.get('iam_roles') if module.params.get('iam_roles') else []
    existing_roles = instance.get('associated_roles', [])
    roles_to_add, roles_to_remove = compare_iam_roles(existing_roles, target_roles, purge_iam_roles)
    if bool(roles_to_add or roles_to_remove):
        changed = True
        if module.check_mode:
            module.exit_json(changed=changed, **instance)
        else:
            update_iam_roles(client, module, instance_id, roles_to_add, roles_to_remove)
    return changed