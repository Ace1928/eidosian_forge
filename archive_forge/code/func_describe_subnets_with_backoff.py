from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
@AWSRetry.exponential_backoff()
def describe_subnets_with_backoff(connection, subnet_ids, filters):
    """
    Describe Subnets with AWSRetry backoff throttling support.

    connection  : boto3 client connection object
    subnet_ids  : list of subnet ids for which to gather information
    filters     : additional filters to apply to request
    """
    return connection.describe_subnets(SubnetIds=subnet_ids, Filters=filters)