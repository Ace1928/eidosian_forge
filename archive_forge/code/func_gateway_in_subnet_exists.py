import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def gateway_in_subnet_exists(client, module, subnet_id, allocation_id=None):
    """Retrieve all NAT Gateways for a subnet.
    Args:
        client (botocore.client.EC2): Boto3 client
        module: AnsibleAWSModule class instance
        subnet_id (str): The subnet_id the nat resides in.

    Kwargs:
        allocation_id (str): The EIP Amazon identifier.
            default = None

    Basic Usage:
        >>> client = boto3.client('ec2')
        >>> module = AnsibleAWSModule(...)
        >>> subnet_id = 'subnet-1234567'
        >>> allocation_id = 'eipalloc-1234567'
        >>> gateway_in_subnet_exists(client, module, subnet_id, allocation_id)
        (
            [
                {
                    "create_time": "2016-03-05T00:33:21.209000+00:00",
                    "delete_time": "2016-03-05T00:36:37.329000+00:00",
                    "nat_gateway_addresses": [
                        {
                            "public_ip": "55.55.55.55",
                            "network_interface_id": "eni-1234567",
                            "private_ip": "10.0.0.102",
                            "allocation_id": "eipalloc-1234567"
                        }
                    ],
                    "nat_gateway_id": "nat-123456789",
                    "state": "deleted",
                    "subnet_id": "subnet-123456789",
                    "tags": {},
                    "vpc_id": "vpc-1234567"
                }
            ],
            False
        )

    Returns:
        Tuple (list, bool)
    """
    allocation_id_exists = False
    gateways = []
    states = ['available', 'pending']
    gws_retrieved = get_nat_gateways(client, module, subnet_id, states=states)
    if gws_retrieved:
        for gw in gws_retrieved:
            for address in gw['nat_gateway_addresses']:
                if allocation_id:
                    if address.get('allocation_id') == allocation_id:
                        allocation_id_exists = True
                        gateways.append(gw)
                else:
                    gateways.append(gw)
    return (gateways, allocation_id_exists)