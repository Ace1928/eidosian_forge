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
@staticmethod
def build_create_image_parameters(**kwargs):
    architecture = kwargs.get('architecture')
    billing_products = kwargs.get('billing_products')
    boot_mode = kwargs.get('boot_mode')
    description = kwargs.get('description')
    device_mapping = kwargs.get('device_mapping') or []
    enhanced_networking = kwargs.get('enhanced_networking')
    image_location = kwargs.get('image_location')
    instance_id = kwargs.get('instance_id')
    kernel_id = kwargs.get('kernel_id')
    name = kwargs.get('name')
    no_reboot = kwargs.get('no_reboot')
    ramdisk_id = kwargs.get('ramdisk_id')
    root_device_name = kwargs.get('root_device_name')
    sriov_net_support = kwargs.get('sriov_net_support')
    tags = kwargs.get('tags')
    tpm_support = kwargs.get('tpm_support')
    uefi_data = kwargs.get('uefi_data')
    virtualization_type = kwargs.get('virtualization_type')
    params = {'Name': name, 'Description': description, 'BlockDeviceMappings': CreateImage.build_block_device_mapping(device_mapping)}
    if instance_id:
        params.update({'InstanceId': instance_id, 'NoReboot': no_reboot, 'TagSpecifications': boto3_tag_specifications(tags, types=['image', 'snapshot'])})
    else:
        params.update({'Architecture': architecture, 'BillingProducts': billing_products, 'BootMode': boot_mode, 'EnaSupport': enhanced_networking, 'ImageLocation': image_location, 'KernelId': kernel_id, 'RamdiskId': ramdisk_id, 'RootDeviceName': root_device_name, 'SriovNetSupport': sriov_net_support, 'TpmSupport': tpm_support, 'UefiData': uefi_data, 'VirtualizationType': virtualization_type})
    return {k: v for k, v in params.items() if v}