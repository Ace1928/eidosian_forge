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
def build_run_instance_spec(params, current_count=0):
    spec = dict(ClientToken=uuid.uuid4().hex, MaxCount=1, MinCount=1)
    spec.update(**build_top_level_options(params))
    spec['NetworkInterfaces'] = build_network_spec(params)
    spec['BlockDeviceMappings'] = build_volume_spec(params)
    tag_spec = build_instance_tags(params)
    if tag_spec is not None:
        spec['TagSpecifications'] = tag_spec
    if params.get('iam_instance_profile'):
        spec['IamInstanceProfile'] = dict(Arn=determine_iam_role(params.get('iam_instance_profile')))
    if params.get('exact_count'):
        spec['MaxCount'] = params.get('exact_count') - current_count
        spec['MinCount'] = params.get('exact_count') - current_count
    if params.get('count'):
        spec['MaxCount'] = params.get('count')
        spec['MinCount'] = params.get('count')
    if params.get('instance_type'):
        spec['InstanceType'] = params['instance_type']
    if not (params.get('instance_type') or params.get('launch_template')):
        raise Ec2InstanceAWSError("At least one of 'instance_type' and 'launch_template' must be passed when launching instances.")
    return spec