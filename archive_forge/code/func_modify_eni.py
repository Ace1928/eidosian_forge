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
def modify_eni(connection, module, eni):
    instance_id = module.params.get('instance_id')
    attached = module.params.get('attached')
    device_index = module.params.get('device_index')
    description = module.params.get('description')
    security_groups = module.params.get('security_groups')
    source_dest_check = module.params.get('source_dest_check')
    delete_on_termination = module.params.get('delete_on_termination')
    secondary_private_ip_addresses = module.params.get('secondary_private_ip_addresses')
    purge_secondary_private_ip_addresses = module.params.get('purge_secondary_private_ip_addresses')
    secondary_private_ip_address_count = module.params.get('secondary_private_ip_address_count')
    allow_reassignment = module.params.get('allow_reassignment')
    changed = False
    tags = module.params.get('tags')
    name = module.params.get('name')
    purge_tags = module.params.get('purge_tags')
    eni = uniquely_find_eni(connection, module, eni)
    eni_id = eni['NetworkInterfaceId']
    try:
        if description is not None:
            if 'Description' not in eni or eni['Description'] != description:
                if not module.check_mode:
                    connection.modify_network_interface_attribute(aws_retry=True, NetworkInterfaceId=eni_id, Description={'Value': description})
                changed = True
        if len(security_groups) > 0:
            groups = get_ec2_security_group_ids_from_names(security_groups, connection, vpc_id=eni['VpcId'], boto3=True)
            if sorted(get_sec_group_list(eni['Groups'])) != sorted(groups):
                if not module.check_mode:
                    connection.modify_network_interface_attribute(aws_retry=True, NetworkInterfaceId=eni_id, Groups=groups)
                changed = True
        if source_dest_check is not None:
            if 'SourceDestCheck' not in eni or eni['SourceDestCheck'] != source_dest_check:
                if not module.check_mode:
                    connection.modify_network_interface_attribute(aws_retry=True, NetworkInterfaceId=eni_id, SourceDestCheck={'Value': source_dest_check})
                changed = True
        if delete_on_termination is not None and 'Attachment' in eni:
            if eni['Attachment']['DeleteOnTermination'] is not delete_on_termination:
                if not module.check_mode:
                    connection.modify_network_interface_attribute(aws_retry=True, NetworkInterfaceId=eni_id, Attachment={'AttachmentId': eni['Attachment']['AttachmentId'], 'DeleteOnTermination': delete_on_termination})
                    if delete_on_termination:
                        waiter = 'network_interface_delete_on_terminate'
                    else:
                        waiter = 'network_interface_no_delete_on_terminate'
                    get_waiter(connection, waiter).wait(NetworkInterfaceIds=[eni_id])
                changed = True
        current_secondary_addresses = []
        if 'PrivateIpAddresses' in eni:
            current_secondary_addresses = [i['PrivateIpAddress'] for i in eni['PrivateIpAddresses'] if not i['Primary']]
        if secondary_private_ip_addresses is not None:
            secondary_addresses_to_remove = list(set(current_secondary_addresses) - set(secondary_private_ip_addresses))
            if secondary_addresses_to_remove and purge_secondary_private_ip_addresses:
                if not module.check_mode:
                    connection.unassign_private_ip_addresses(aws_retry=True, NetworkInterfaceId=eni_id, PrivateIpAddresses=list(set(current_secondary_addresses) - set(secondary_private_ip_addresses)))
                    wait_for(absent_ips, connection, secondary_addresses_to_remove, module, eni_id)
                changed = True
            secondary_addresses_to_add = list(set(secondary_private_ip_addresses) - set(current_secondary_addresses))
            if secondary_addresses_to_add:
                if not module.check_mode:
                    connection.assign_private_ip_addresses(aws_retry=True, NetworkInterfaceId=eni_id, PrivateIpAddresses=secondary_addresses_to_add, AllowReassignment=allow_reassignment)
                    wait_for(correct_ips, connection, secondary_addresses_to_add, module, eni_id)
                changed = True
        if secondary_private_ip_address_count is not None:
            current_secondary_address_count = len(current_secondary_addresses)
            if secondary_private_ip_address_count > current_secondary_address_count:
                if not module.check_mode:
                    connection.assign_private_ip_addresses(aws_retry=True, NetworkInterfaceId=eni_id, SecondaryPrivateIpAddressCount=secondary_private_ip_address_count - current_secondary_address_count, AllowReassignment=allow_reassignment)
                    wait_for(correct_ip_count, connection, secondary_private_ip_address_count, module, eni_id)
                changed = True
            elif secondary_private_ip_address_count < current_secondary_address_count:
                if not module.check_mode:
                    secondary_addresses_to_remove_count = current_secondary_address_count - secondary_private_ip_address_count
                    connection.unassign_private_ip_addresses(aws_retry=True, NetworkInterfaceId=eni_id, PrivateIpAddresses=current_secondary_addresses[:secondary_addresses_to_remove_count])
                    wait_for(correct_ip_count, connection, secondary_private_ip_address_count, module, eni_id)
                changed = True
        if attached is True:
            if 'Attachment' in eni and eni['Attachment']['InstanceId'] != instance_id:
                if not module.check_mode:
                    detach_eni(connection, eni, module)
                    connection.attach_network_interface(aws_retry=True, InstanceId=instance_id, DeviceIndex=device_index, NetworkInterfaceId=eni_id)
                    get_waiter(connection, 'network_interface_attached').wait(NetworkInterfaceIds=[eni_id])
                changed = True
            if 'Attachment' not in eni:
                if not module.check_mode:
                    connection.attach_network_interface(aws_retry=True, InstanceId=instance_id, DeviceIndex=device_index, NetworkInterfaceId=eni_id)
                    get_waiter(connection, 'network_interface_attached').wait(NetworkInterfaceIds=[eni_id])
                changed = True
        elif attached is False:
            changed |= detach_eni(connection, eni, module)
            get_waiter(connection, 'network_interface_available').wait(NetworkInterfaceIds=[eni_id])
        changed |= manage_tags(connection, module, eni, name, tags, purge_tags)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, f'Failed to modify eni {eni_id}')
    eni = describe_eni(connection, module, eni_id)
    if module.check_mode and changed:
        module.exit_json(changed=changed, msg=f'Would have modified ENI: {eni['NetworkInterfaceId']} if not in check mode')
    module.exit_json(changed=changed, interface=get_eni_info(eni))