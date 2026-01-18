from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def normalize_route_table(table):
    table['tags'] = boto3_tag_list_to_ansible_dict(table['Tags'])
    table['Associations'] = [normalize_association(assoc) for assoc in table['Associations']]
    table['Routes'] = [normalize_route(route) for route in table['Routes']]
    table['Id'] = table['RouteTableId']
    del table['Tags']
    return camel_dict_to_snake_dict(table, ignore_list=['tags'])