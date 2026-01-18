from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def get_internet_gateway_info(internet_gateway, convert_tags):
    if convert_tags:
        tags = boto3_tag_list_to_ansible_dict(internet_gateway['Tags'])
        ignore_list = ['Tags']
    else:
        tags = internet_gateway['Tags']
        ignore_list = []
    internet_gateway_info = {'InternetGatewayId': internet_gateway['InternetGatewayId'], 'Attachments': internet_gateway['Attachments'], 'Tags': tags}
    internet_gateway_info = camel_dict_to_snake_dict(internet_gateway_info, ignore_list=ignore_list)
    return internet_gateway_info