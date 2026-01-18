import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def pre_create(client, module, subnet_id, tags, purge_tags, allocation_id=None, eip_address=None, if_exist_do_not_create=False, wait=False, client_token=None, connectivity_type='public', default_create=False):
    """Create an Amazon NAT Gateway.
    Args:
        client (botocore.client.EC2): Boto3 client
        module: AnsibleAWSModule class instance
        subnet_id (str): The subnet_id the nat resides in
        tags (dict): Tags to associate to the NAT gateway
        purge_tags (bool): If true, remove tags not listed in I(tags)

    Kwargs:
        allocation_id (str): The EIP Amazon identifier.
            default = None
        eip_address (str): The Elastic IP Address of the EIP.
            default = None
        if_exist_do_not_create (bool): if a nat gateway already exists in this
            subnet, than do not create another one.
            default = False
        wait (bool): Wait for the nat to be in the deleted state before returning.
            default = False
        client_token (str):
            default = None
        default_create (bool): create a NAT gateway even if EIP address is not found.
            default = False

    Basic Usage:
        >>> client = boto3.client('ec2')
        >>> module = AnsibleAWSModule(...)
        >>> subnet_id = 'subnet-w4t12897'
        >>> allocation_id = 'eipalloc-36014da3'
        >>> pre_create(client, module, subnet_id, allocation_id, if_exist_do_not_create=True, wait=True, connectivity_type=public)
        [
            true,
            "",
            {
                "create_time": "2016-03-05T00:33:21.209000+00:00",
                "delete_time": "2016-03-05T00:36:37.329000+00:00",
                "nat_gateway_addresses": [
                    {
                        "public_ip": "52.87.29.36",
                        "network_interface_id": "eni-5579742d",
                        "private_ip": "10.0.0.102",
                        "allocation_id": "eipalloc-36014da3"
                    }
                ],
                "nat_gateway_id": "nat-03835afb6e31df79b",
                "state": "deleted",
                "subnet_id": "subnet-w4t12897",
                "tags": {},
                "vpc_id": "vpc-w68571b5"
            }
        ]

    Returns:
        Tuple (bool, str, list)
    """
    changed = False
    msg = ''
    results = {}
    if not allocation_id and (not eip_address):
        existing_gateways, allocation_id_exists = gateway_in_subnet_exists(client, module, subnet_id)
        if len(existing_gateways) > 0 and if_exist_do_not_create:
            results = existing_gateways[0]
            changed |= ensure_ec2_tags(client, module, results['nat_gateway_id'], resource_type='natgateway', tags=tags, purge_tags=purge_tags)
            results['tags'] = describe_ec2_tags(client, module, results['nat_gateway_id'], resource_type='natgateway')
            if changed:
                return (changed, msg, results)
            changed = False
            msg = f'NAT Gateway {existing_gateways[0]['nat_gateway_id']} already exists in subnet_id {subnet_id}'
            return (changed, msg, results)
        elif connectivity_type == 'public':
            changed, msg, allocation_id = allocate_eip_address(client, module)
            if not changed:
                return (changed, msg, dict())
    elif eip_address or allocation_id:
        if eip_address and (not allocation_id):
            allocation_id, msg = get_eip_allocation_id_by_address(client, module, eip_address)
            if not allocation_id and (not default_create):
                changed = False
                module.fail_json(msg=msg)
            elif not allocation_id and default_create:
                eip_address = None
                return pre_create(client, module, subnet_id, tags, purge_tags, allocation_id, eip_address, if_exist_do_not_create, wait, client_token, connectivity_type, default_create)
        existing_gateways, allocation_id_exists = gateway_in_subnet_exists(client, module, subnet_id, allocation_id)
        if len(existing_gateways) > 0 and (allocation_id_exists or if_exist_do_not_create):
            results = existing_gateways[0]
            changed |= ensure_ec2_tags(client, module, results['nat_gateway_id'], resource_type='natgateway', tags=tags, purge_tags=purge_tags)
            results['tags'] = describe_ec2_tags(client, module, results['nat_gateway_id'], resource_type='natgateway')
            if changed:
                return (changed, msg, results)
            changed = False
            msg = f'NAT Gateway {existing_gateways[0]['nat_gateway_id']} already exists in subnet_id {subnet_id}'
            return (changed, msg, results)
    changed, results, msg = create(client, module, subnet_id, allocation_id, tags, client_token, wait, connectivity_type)
    return (changed, msg, results)