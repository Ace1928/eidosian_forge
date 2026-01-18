import time
from ipaddress import ip_address
from ipaddress import ip_network
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def create_eni(connection, vpc_id, module):
    instance_id = module.params.get('instance_id')
    attached = module.params.get('attached')
    if instance_id == 'None':
        instance_id = None
    device_index = module.params.get('device_index')
    subnet_id = module.params.get('subnet_id')
    private_ip_address = module.params.get('private_ip_address')
    description = module.params.get('description')
    security_groups = get_ec2_security_group_ids_from_names(module.params.get('security_groups'), connection, vpc_id=vpc_id, boto3=True)
    secondary_private_ip_addresses = module.params.get('secondary_private_ip_addresses')
    secondary_private_ip_address_count = module.params.get('secondary_private_ip_address_count')
    changed = False
    tags = module.params.get('tags') or dict()
    name = module.params.get('name')
    if name:
        tags['Name'] = name
    try:
        args = {'SubnetId': subnet_id}
        if private_ip_address:
            args['PrivateIpAddress'] = private_ip_address
        if description:
            args['Description'] = description
        if len(security_groups) > 0:
            args['Groups'] = security_groups
        if tags:
            args['TagSpecifications'] = boto3_tag_specifications(tags, types='network-interface')
        if private_ip_address:
            cidr_block = connection.describe_subnets(SubnetIds=[str(subnet_id)])['Subnets'][0]['CidrBlock']
            valid_private_ip = ip_address(private_ip_address) in ip_network(cidr_block)
            if not valid_private_ip:
                module.fail_json(changed=False, msg="Error: cannot create ENI - Address does not fall within the subnet's address range.")
        if module.check_mode:
            module.exit_json(changed=True, msg='Would have created ENI if not in check mode.')
        eni_dict = connection.create_network_interface(aws_retry=True, **args)
        eni = eni_dict['NetworkInterface']
        eni_id = eni['NetworkInterfaceId']
        get_waiter(connection, 'network_interface_available').wait(NetworkInterfaceIds=[eni_id])
        if attached and instance_id is not None:
            try:
                connection.attach_network_interface(aws_retry=True, InstanceId=instance_id, DeviceIndex=device_index, NetworkInterfaceId=eni['NetworkInterfaceId'])
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError):
                connection.delete_network_interface(aws_retry=True, NetworkInterfaceId=eni_id)
                raise
            get_waiter(connection, 'network_interface_attached').wait(NetworkInterfaceIds=[eni_id])
        if secondary_private_ip_address_count is not None:
            try:
                connection.assign_private_ip_addresses(aws_retry=True, NetworkInterfaceId=eni['NetworkInterfaceId'], SecondaryPrivateIpAddressCount=secondary_private_ip_address_count)
                wait_for(correct_ip_count, connection, secondary_private_ip_address_count, module, eni_id)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError):
                connection.delete_network_interface(aws_retry=True, NetworkInterfaceId=eni_id)
                raise
        if secondary_private_ip_addresses is not None:
            try:
                connection.assign_private_ip_addresses(NetworkInterfaceId=eni['NetworkInterfaceId'], PrivateIpAddresses=secondary_private_ip_addresses)
                wait_for(correct_ips, connection, secondary_private_ip_addresses, module, eni_id)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError):
                connection.delete_network_interface(aws_retry=True, NetworkInterfaceId=eni_id)
                raise
        eni = describe_eni(connection, module, eni_id)
        changed = True
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, f'Failed to create eni {name} for {subnet_id} in {vpc_id} with {private_ip_address}')
    module.exit_json(changed=changed, interface=get_eni_info(eni))