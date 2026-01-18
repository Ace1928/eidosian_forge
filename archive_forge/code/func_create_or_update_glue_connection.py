import copy
import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_glue_connection(connection, connection_ec2, module, glue_connection):
    """
    Create or update an AWS Glue connection

    :param connection: AWS boto3 glue connection
    :param module: Ansible module
    :param glue_connection: a dict of AWS Glue connection parameters or None
    :return:
    """
    changed = False
    params = dict()
    params['ConnectionInput'] = dict()
    params['ConnectionInput']['Name'] = module.params.get('name')
    params['ConnectionInput']['ConnectionType'] = module.params.get('connection_type')
    params['ConnectionInput']['ConnectionProperties'] = module.params.get('connection_properties')
    if module.params.get('catalog_id') is not None:
        params['CatalogId'] = module.params.get('catalog_id')
    if module.params.get('description') is not None:
        params['ConnectionInput']['Description'] = module.params.get('description')
    if module.params.get('match_criteria') is not None:
        params['ConnectionInput']['MatchCriteria'] = module.params.get('match_criteria')
    if module.params.get('security_groups') is not None or module.params.get('subnet_id') is not None:
        params['ConnectionInput']['PhysicalConnectionRequirements'] = dict()
    if module.params.get('security_groups') is not None:
        security_group_ids = get_ec2_security_group_ids_from_names(module.params.get('security_groups'), connection_ec2, boto3=True)
        params['ConnectionInput']['PhysicalConnectionRequirements']['SecurityGroupIdList'] = security_group_ids
    if module.params.get('subnet_id') is not None:
        params['ConnectionInput']['PhysicalConnectionRequirements']['SubnetId'] = module.params.get('subnet_id')
    if module.params.get('availability_zone') is not None:
        params['ConnectionInput']['PhysicalConnectionRequirements']['AvailabilityZone'] = module.params.get('availability_zone')
    if glue_connection:
        if _compare_glue_connection_params(params, glue_connection):
            try:
                update_params = copy.deepcopy(params)
                update_params['Name'] = update_params['ConnectionInput']['Name']
                if not module.check_mode:
                    connection.update_connection(aws_retry=True, **update_params)
                changed = True
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e)
    else:
        try:
            if not module.check_mode:
                connection.create_connection(aws_retry=True, **params)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e)
    if changed and (not module.check_mode):
        glue_connection = _await_glue_connection(connection, module)
    if glue_connection:
        module.deprecate("The 'connection_properties' return key is deprecated and will be replaced by 'raw_connection_properties'. Both values are returned for now.", date='2024-06-01', collection_name='community.aws')
        glue_connection['RawConnectionProperties'] = glue_connection['ConnectionProperties']
    module.exit_json(changed=changed, **camel_dict_to_snake_dict(glue_connection or {}, ignore_list=['RawConnectionProperties']))