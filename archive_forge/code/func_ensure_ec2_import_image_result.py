import_image:
import copy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import helper_describe_import_image_tasks
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def ensure_ec2_import_image_result(import_image_info):
    result = {'import_image': {}}
    if import_image_info:
        image = copy.deepcopy(import_image_info[0])
        image['Tags'] = boto3_tag_list_to_ansible_dict(image['Tags'])
        result['import_image'] = camel_dict_to_snake_dict(image, ignore_list=['Tags'])
    return result