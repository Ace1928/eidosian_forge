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
def build_volume_spec(params):
    volumes = params.get('volumes') or []
    for volume in volumes:
        if 'ebs' in volume:
            for int_value in ['volume_size', 'iops']:
                if int_value in volume['ebs']:
                    volume['ebs'][int_value] = int(volume['ebs'][int_value])
            if 'volume_type' in volume['ebs'] and volume['ebs']['volume_type'] == 'gp3':
                if not volume['ebs'].get('iops'):
                    volume['ebs']['iops'] = 3000
                if 'throughput' in volume['ebs']:
                    volume['ebs']['throughput'] = int(volume['ebs']['throughput'])
                else:
                    volume['ebs']['throughput'] = 125
    return [snake_dict_to_camel_dict(v, capitalize_first=True) for v in volumes]