import re
from ipaddress import ip_network
from time import sleep
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def find_subnets(connection, module, vpc_id, identified_subnets):
    """
    Finds a list of subnets, each identified either by a raw ID, a unique
    'Name' tag, or a CIDR such as 10.0.0.0/8.
    """
    CIDR_RE = re.compile('^(\\d{1,3}\\.){3}\\d{1,3}/\\d{1,2}$')
    SUBNET_RE = re.compile('^subnet-[A-z0-9]+$')
    subnet_ids = []
    subnet_names = []
    subnet_cidrs = []
    for subnet in identified_subnets or []:
        if re.match(SUBNET_RE, subnet):
            subnet_ids.append(subnet)
        elif re.match(CIDR_RE, subnet):
            subnet_cidrs.append(subnet)
        else:
            subnet_names.append(subnet)
    subnets_by_id = []
    if subnet_ids:
        filters = ansible_dict_to_boto3_filter_list({'vpc-id': vpc_id})
        try:
            subnets_by_id = describe_subnets_with_backoff(connection, SubnetIds=subnet_ids, Filters=filters)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f"Couldn't find subnet with id {subnet_ids}")
    subnets_by_cidr = []
    if subnet_cidrs:
        filters = ansible_dict_to_boto3_filter_list({'vpc-id': vpc_id, 'cidr': subnet_cidrs})
        try:
            subnets_by_cidr = describe_subnets_with_backoff(connection, Filters=filters)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f"Couldn't find subnet with cidr {subnet_cidrs}")
    subnets_by_name = []
    if subnet_names:
        filters = ansible_dict_to_boto3_filter_list({'vpc-id': vpc_id, 'tag:Name': subnet_names})
        try:
            subnets_by_name = describe_subnets_with_backoff(connection, Filters=filters)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f"Couldn't find subnet with names {subnet_names}")
        for name in subnet_names:
            matching_count = len([1 for s in subnets_by_name for t in s.get('Tags', []) if t['Key'] == 'Name' and t['Value'] == name])
            if matching_count == 0:
                module.fail_json(msg=f'Subnet named "{name}" does not exist')
            elif matching_count > 1:
                module.fail_json(msg=f'Multiple subnets named "{name}"')
    return subnets_by_id + subnets_by_cidr + subnets_by_name