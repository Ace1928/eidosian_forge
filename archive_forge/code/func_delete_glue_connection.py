import copy
import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_glue_connection(connection, module, glue_connection):
    """
    Delete an AWS Glue connection

    :param connection: AWS boto3 glue connection
    :param module: Ansible module
    :param glue_connection: a dict of AWS Glue connection parameters or None
    :return:
    """
    changed = False
    params = {'ConnectionName': module.params.get('name')}
    if module.params.get('catalog_id') is not None:
        params['CatalogId'] = module.params.get('catalog_id')
    if glue_connection:
        try:
            if not module.check_mode:
                connection.delete_connection(aws_retry=True, **params)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e)
    module.exit_json(changed=changed)