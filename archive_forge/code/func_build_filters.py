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
def build_filters():
    filters = {'instance-state-name': ['pending', 'running', 'stopping', 'stopped']}
    if isinstance(module.params.get('instance_ids'), string_types):
        filters['instance-id'] = [module.params.get('instance_ids')]
    elif isinstance(module.params.get('instance_ids'), list) and len(module.params.get('instance_ids')):
        filters['instance-id'] = module.params.get('instance_ids')
    else:
        if not module.params.get('vpc_subnet_id'):
            if module.params.get('network'):
                ints = module.params.get('network').get('interfaces')
                if ints:
                    filters['network-interface.network-interface-id'] = []
                    for i in ints:
                        if isinstance(i, dict):
                            i = i['id']
                        filters['network-interface.network-interface-id'].append(i)
            else:
                sub = get_default_subnet(get_default_vpc(), availability_zone=module.params.get('availability_zone'))
                filters['subnet-id'] = sub['SubnetId']
        else:
            filters['subnet-id'] = [module.params.get('vpc_subnet_id')]
        if module.params.get('name'):
            filters['tag:Name'] = [module.params.get('name')]
        elif module.params.get('tags'):
            name_tag = module.params.get('tags').get('Name', None)
            if name_tag:
                filters['tag:Name'] = [name_tag]
        if module.params.get('image_id'):
            filters['image-id'] = [module.params.get('image_id')]
        elif (module.params.get('image') or {}).get('id'):
            filters['image-id'] = [module.params.get('image', {}).get('id')]
    return filters