import traceback
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_block_device_meta(module, volume):
    if 'snapshot' not in volume and 'ephemeral' not in volume and ('no_device' not in volume):
        if 'volume_size' not in volume:
            module.fail_json(msg='Size must be specified when creating a new volume or modifying the root volume')
    if 'snapshot' in volume:
        if volume.get('volume_type') == 'io1' and 'iops' not in volume:
            module.fail_json(msg='io1 volumes must have an iops value set')
    if 'ephemeral' in volume:
        if 'snapshot' in volume:
            module.fail_json(msg='Cannot set both ephemeral and snapshot')
    return_object = {}
    if 'ephemeral' in volume:
        return_object['VirtualName'] = volume.get('ephemeral')
    if 'device_name' in volume:
        return_object['DeviceName'] = volume.get('device_name')
    if 'no_device' in volume:
        return_object['NoDevice'] = volume.get('no_device')
    if any((key in volume for key in ['snapshot', 'volume_size', 'volume_type', 'delete_on_termination', 'iops', 'throughput', 'encrypted'])):
        return_object['Ebs'] = {}
    if 'snapshot' in volume:
        return_object['Ebs']['SnapshotId'] = volume.get('snapshot')
    if 'volume_size' in volume:
        return_object['Ebs']['VolumeSize'] = int(volume.get('volume_size', 0))
    if 'volume_type' in volume:
        return_object['Ebs']['VolumeType'] = volume.get('volume_type')
    if 'delete_on_termination' in volume:
        return_object['Ebs']['DeleteOnTermination'] = volume.get('delete_on_termination', False)
    if 'iops' in volume:
        return_object['Ebs']['Iops'] = volume.get('iops')
    if 'throughput' in volume:
        if volume.get('volume_type') != 'gp3':
            module.fail_json(msg='The throughput parameter is supported only for GP3 volumes.')
        return_object['Ebs']['Throughput'] = volume.get('throughput')
    if 'encrypted' in volume:
        return_object['Ebs']['Encrypted'] = volume.get('encrypted')
    return return_object