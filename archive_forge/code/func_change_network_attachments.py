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
def change_network_attachments(instance, params):
    if (params.get('network') or {}).get('interfaces') is not None:
        new_ids = []
        for inty in params.get('network').get('interfaces'):
            if isinstance(inty, dict) and 'id' in inty:
                new_ids.append(inty['id'])
            elif isinstance(inty, string_types):
                new_ids.append(inty)
        old_ids = [inty['NetworkInterfaceId'] for inty in instance['NetworkInterfaces']]
        to_attach = set(new_ids) - set(old_ids)
        if not module.check_mode:
            for eni_id in to_attach:
                try:
                    client.attach_network_interface(aws_retry=True, DeviceIndex=new_ids.index(eni_id), InstanceId=instance['InstanceId'], NetworkInterfaceId=eni_id)
                except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                    module.fail_json_aws(e, msg=f'Could not attach interface {eni_id} to instance {instance['InstanceId']}')
        return bool(len(to_attach))
    return False