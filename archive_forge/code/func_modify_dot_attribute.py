import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import is_outpost_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def modify_dot_attribute(module, ec2_conn, instance_dict, device_name):
    """Modify delete_on_termination attribute"""
    delete_on_termination = module.params.get('delete_on_termination')
    changed = False
    mapped_block_device = None
    _attempt = 0
    while mapped_block_device is None:
        _attempt += 1
        instance_dict = get_instance(module, ec2_conn=ec2_conn, instance_id=instance_dict['instance_id'])
        mapped_block_device = get_mapped_block_device(instance_dict=instance_dict, device_name=device_name)
        if mapped_block_device is None:
            if _attempt > 2:
                module.fail_json(msg='Unable to find device on instance', device=device_name, instance=instance_dict)
            time.sleep(1)
    if delete_on_termination != mapped_block_device['ebs'].get('delete_on_termination'):
        try:
            ec2_conn.modify_instance_attribute(aws_retry=True, InstanceId=instance_dict['instance_id'], BlockDeviceMappings=[{'DeviceName': device_name, 'Ebs': {'DeleteOnTermination': delete_on_termination}}])
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f'Error while modifying Block Device Mapping of instance {instance_dict['instance_id']}')
    return changed