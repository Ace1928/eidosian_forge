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
def build_top_level_options(params):
    spec = {}
    if params.get('image_id'):
        spec['ImageId'] = params['image_id']
    elif isinstance(params.get('image'), dict):
        image = params.get('image', {})
        spec['ImageId'] = image.get('id')
        if 'ramdisk' in image:
            spec['RamdiskId'] = image['ramdisk']
        if 'kernel' in image:
            spec['KernelId'] = image['kernel']
    if not spec.get('ImageId') and (not params.get('launch_template')):
        module.fail_json(msg='You must include an image_id or image.id parameter to create an instance, or use a launch_template.')
    if params.get('key_name') is not None:
        spec['KeyName'] = params.get('key_name')
    spec.update(build_userdata(params))
    if params.get('launch_template') is not None:
        spec['LaunchTemplate'] = {}
        if not params.get('launch_template').get('id') and (not params.get('launch_template').get('name')):
            module.fail_json(msg='Could not create instance with launch template. Either launch_template.name or launch_template.id parameters are required')
        if params.get('launch_template').get('id') is not None:
            spec['LaunchTemplate']['LaunchTemplateId'] = params.get('launch_template').get('id')
        if params.get('launch_template').get('name') is not None:
            spec['LaunchTemplate']['LaunchTemplateName'] = params.get('launch_template').get('name')
        if params.get('launch_template').get('version') is not None:
            spec['LaunchTemplate']['Version'] = to_native(params.get('launch_template').get('version'))
    if params.get('detailed_monitoring', False):
        spec['Monitoring'] = {'Enabled': True}
    if params.get('cpu_credit_specification') is not None:
        spec['CreditSpecification'] = {'CpuCredits': params.get('cpu_credit_specification')}
    if params.get('tenancy') is not None:
        spec['Placement'] = {'Tenancy': params.get('tenancy')}
    if params.get('placement_group'):
        if 'Placement' in spec:
            spec['Placement']['GroupName'] = str(params.get('placement_group'))
        else:
            spec.setdefault('Placement', {'GroupName': str(params.get('placement_group'))})
    if params.get('placement') is not None:
        spec['Placement'] = {}
        if params.get('placement').get('availability_zone') is not None:
            spec['Placement']['AvailabilityZone'] = params.get('placement').get('availability_zone')
        if params.get('placement').get('affinity') is not None:
            spec['Placement']['Affinity'] = params.get('placement').get('affinity')
        if params.get('placement').get('group_name') is not None:
            spec['Placement']['GroupName'] = params.get('placement').get('group_name')
        if params.get('placement').get('host_id') is not None:
            spec['Placement']['HostId'] = params.get('placement').get('host_id')
        if params.get('placement').get('host_resource_group_arn') is not None:
            spec['Placement']['HostResourceGroupArn'] = params.get('placement').get('host_resource_group_arn')
        if params.get('placement').get('partition_number') is not None:
            spec['Placement']['PartitionNumber'] = params.get('placement').get('partition_number')
        if params.get('placement').get('tenancy') is not None:
            spec['Placement']['Tenancy'] = params.get('placement').get('tenancy')
    if params.get('ebs_optimized') is not None:
        spec['EbsOptimized'] = params.get('ebs_optimized')
    if params.get('instance_initiated_shutdown_behavior'):
        spec['InstanceInitiatedShutdownBehavior'] = params.get('instance_initiated_shutdown_behavior')
    if params.get('termination_protection') is not None:
        spec['DisableApiTermination'] = params.get('termination_protection')
    if params.get('hibernation_options') and params.get('volumes'):
        for vol in params['volumes']:
            if vol.get('ebs') and vol['ebs'].get('encrypted'):
                spec['HibernationOptions'] = {'Configured': True}
            else:
                module.fail_json(msg='Hibernation prerequisites not satisfied. Refer to https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/hibernating-prerequisites.html')
    if params.get('cpu_options') is not None:
        spec['CpuOptions'] = {}
        spec['CpuOptions']['ThreadsPerCore'] = params.get('cpu_options').get('threads_per_core')
        spec['CpuOptions']['CoreCount'] = params.get('cpu_options').get('core_count')
    if params.get('metadata_options'):
        spec['MetadataOptions'] = {}
        spec['MetadataOptions']['HttpEndpoint'] = params.get('metadata_options').get('http_endpoint')
        spec['MetadataOptions']['HttpTokens'] = params.get('metadata_options').get('http_tokens')
        spec['MetadataOptions']['HttpPutResponseHopLimit'] = params.get('metadata_options').get('http_put_response_hop_limit')
        spec['MetadataOptions']['HttpProtocolIpv6'] = params.get('metadata_options').get('http_protocol_ipv6')
        spec['MetadataOptions']['InstanceMetadataTags'] = params.get('metadata_options').get('instance_metadata_tags')
    if params.get('additional_info'):
        spec['AdditionalInfo'] = params.get('additional_info')
    if params.get('license_specifications'):
        spec['LicenseSpecifications'] = []
        for license_configuration in params.get('license_specifications'):
            spec['LicenseSpecifications'].append({'LicenseConfigurationArn': license_configuration.get('license_configuration_arn')})
    return spec