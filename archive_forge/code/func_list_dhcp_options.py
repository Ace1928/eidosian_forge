from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import normalize_ec2_vpc_dhcp_config
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def list_dhcp_options(client, module):
    params = dict(Filters=ansible_dict_to_boto3_filter_list(module.params.get('filters')))
    if module.params.get('dry_run'):
        params['DryRun'] = True
    if module.params.get('dhcp_options_ids'):
        params['DhcpOptionsIds'] = module.params.get('dhcp_options_ids')
    try:
        all_dhcp_options = client.describe_dhcp_options(aws_retry=True, **params)
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)
    normalized_config = [normalize_ec2_vpc_dhcp_config(config['DhcpConfigurations']) for config in all_dhcp_options['DhcpOptions']]
    raw_config = [camel_dict_to_snake_dict(get_dhcp_options_info(option), ignore_list=['Tags']) for option in all_dhcp_options['DhcpOptions']]
    return (raw_config, normalized_config)