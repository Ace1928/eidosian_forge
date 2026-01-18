from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def create_subnet_list(subnets):
    """
    Construct a list of subnet ids from a list of subnets dicts returned by boto3.
    Parameters:
        subnets (list): A list of subnets definitions.
        @see https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rds.html#RDS.Client.describe_db_subnet_groups
    Returns:
        (list): List of subnet ids (str)
    """
    subnets_ids = []
    for subnet in subnets:
        subnets_ids.append(subnet.get('subnet_identifier'))
    return subnets_ids