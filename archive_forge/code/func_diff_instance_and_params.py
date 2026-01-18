import time
import uuid
from collections import namedtuple
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tower import tower_callback_script
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def diff_instance_and_params(instance, params, skip=None):
    """boto3 instance obj, module params"""
    if skip is None:
        skip = []
    changes_to_apply = []
    id_ = instance['InstanceId']
    ParamMapper = namedtuple('ParamMapper', ['param_key', 'instance_key', 'attribute_name', 'add_value'])

    def value_wrapper(v):
        return {'Value': v}
    param_mappings = [ParamMapper('ebs_optimized', 'EbsOptimized', 'ebsOptimized', value_wrapper), ParamMapper('termination_protection', 'DisableApiTermination', 'disableApiTermination', value_wrapper)]
    for mapping in param_mappings:
        if params.get(mapping.param_key) is None:
            continue
        if mapping.instance_key in skip:
            continue
        try:
            value = client.describe_instance_attribute(aws_retry=True, Attribute=mapping.attribute_name, InstanceId=id_)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg=f'Could not describe attribute {mapping.attribute_name} for instance {id_}')
        if value[mapping.instance_key]['Value'] != params.get(mapping.param_key):
            arguments = dict(InstanceId=instance['InstanceId'])
            arguments[mapping.instance_key] = mapping.add_value(params.get(mapping.param_key))
            changes_to_apply.append(arguments)
    if params.get('security_group') or params.get('security_groups'):
        try:
            value = client.describe_instance_attribute(aws_retry=True, Attribute='groupSet', InstanceId=id_)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg=f'Could not describe attribute groupSet for instance {id_}')
        if params.get('vpc_subnet_id'):
            subnet_id = params.get('vpc_subnet_id')
        else:
            default_vpc = get_default_vpc()
            if default_vpc is None:
                module.fail_json(msg='No default subnet could be found - you must include a VPC subnet ID (vpc_subnet_id parameter) to modify security groups.')
            else:
                sub = get_default_subnet(default_vpc)
                subnet_id = sub['SubnetId']
        groups = discover_security_groups(group=params.get('security_group'), groups=params.get('security_groups'), subnet_id=subnet_id)
        expected_groups = groups
        instance_groups = [g['GroupId'] for g in value['Groups']]
        if set(instance_groups) != set(expected_groups):
            changes_to_apply.append(dict(Groups=expected_groups, InstanceId=instance['InstanceId']))
    if (params.get('network') or {}).get('source_dest_check') is not None:
        check = bool(params.get('network').get('source_dest_check'))
        if instance['SourceDestCheck'] != check:
            changes_to_apply.append(dict(InstanceId=instance['InstanceId'], SourceDestCheck={'Value': check}))
    return changes_to_apply