import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def describe_instance_attributes(connection, instance_id, attributes):
    result = {}
    for attr in attributes:
        response = connection.describe_instance_attribute(Attribute=attr, InstanceId=instance_id)
        for key in response:
            if key not in ('InstanceId', 'ResponseMetadata'):
                result[key] = response[key]
    return result