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
def discover_security_groups(group, groups, parent_vpc_id=None, subnet_id=None):
    if subnet_id is not None:
        try:
            sub = client.describe_subnets(aws_retry=True, SubnetIds=[subnet_id])
        except is_boto3_error_code('InvalidGroup.NotFound'):
            module.fail_json(f'Could not find subnet {subnet_id} to associate security groups. Please check the vpc_subnet_id and security_groups parameters.')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f'Error while searching for subnet {subnet_id} parent VPC.')
        parent_vpc_id = sub['Subnets'][0]['VpcId']
    if group:
        return get_ec2_security_group_ids_from_names(group, client, vpc_id=parent_vpc_id)
    if groups:
        return get_ec2_security_group_ids_from_names(groups, client, vpc_id=parent_vpc_id)
    return []