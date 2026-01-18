from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import normalize_ec2_vpc_dhcp_config
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
def create_dhcp_option_set(client, module, new_config):
    """
    A CreateDhcpOptions object looks different than the object we create in create_dhcp_config()
    This is the only place we use it, so create it now
    https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateDhcpOptions.html
    We have to do this after inheriting any existing_config, so we need to start with the object
    that we made in create_dhcp_config().
    normalize_config() gives us the nicest format to work with for this.
    """
    changed = True
    desired_config = normalize_ec2_vpc_dhcp_config(new_config)
    create_config = []
    tags_list = []
    for option in ['domain-name', 'domain-name-servers', 'ntp-servers', 'netbios-name-servers']:
        if desired_config.get(option):
            create_config.append({'Key': option, 'Values': desired_config[option]})
    if desired_config.get('netbios-node-type'):
        create_config.append({'Key': 'netbios-node-type', 'Values': [desired_config['netbios-node-type']]})
    if module.params.get('tags'):
        tags_list = boto3_tag_specifications(module.params['tags'], ['dhcp-options'])
    try:
        if not module.check_mode:
            dhcp_options = client.create_dhcp_options(aws_retry=True, DhcpConfigurations=create_config, TagSpecifications=tags_list)
            return (changed, dhcp_options['DhcpOptions']['DhcpOptionsId'])
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Unable to create dhcp option set')
    return (changed, None)