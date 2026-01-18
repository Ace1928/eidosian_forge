import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto3_conn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_aws_connection_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def ordered_obj(obj):
    """
    Order object for comparison purposes

    :param obj:
    :return:
    """
    if isinstance(obj, dict):
        return sorted(((k, ordered_obj(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return sorted((ordered_obj(x) for x in obj))
    else:
        return obj