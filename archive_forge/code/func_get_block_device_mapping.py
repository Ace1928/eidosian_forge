import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import add_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_block_device_mapping(image):
    bdm_dict = {}
    if image is not None and image.get('block_device_mappings') is not None:
        bdm = image.get('block_device_mappings')
        for device in bdm:
            device_name = device.get('device_name')
            if 'ebs' in device:
                ebs = device.get('ebs')
                bdm_dict_item = {'size': ebs.get('volume_size'), 'snapshot_id': ebs.get('snapshot_id'), 'volume_type': ebs.get('volume_type'), 'encrypted': ebs.get('encrypted'), 'delete_on_termination': ebs.get('delete_on_termination')}
            elif 'virtual_name' in device:
                bdm_dict_item = dict(virtual_name=device['virtual_name'])
            bdm_dict[device_name] = bdm_dict_item
    return bdm_dict