import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def assemble_params_for_creating_vi(params):
    """
    Returns kwargs to use in the call to create the virtual interface

    Params for public virtual interfaces:
    virtualInterfaceName, vlan, asn, authKey, amazonAddress, customerAddress, addressFamily, cidr
    Params for private virtual interfaces:
    virtualInterfaceName, vlan, asn, authKey, amazonAddress, customerAddress, addressFamily, virtualGatewayId
    """
    public = params['public']
    name = params['name']
    vlan = params['vlan']
    bgp_asn = params['bgp_asn']
    auth_key = params['authentication_key']
    amazon_addr = params['amazon_address']
    customer_addr = params['customer_address']
    family_addr = params['address_type']
    cidr = params['cidr']
    virtual_gateway_id = params['virtual_gateway_id']
    direct_connect_gateway_id = params['direct_connect_gateway_id']
    parameters = dict(virtualInterfaceName=name, vlan=vlan, asn=bgp_asn)
    opt_params = dict(authKey=auth_key, amazonAddress=amazon_addr, customerAddress=customer_addr, addressFamily=family_addr)
    for name, value in opt_params.items():
        if value:
            parameters[name] = value
    if public and cidr:
        parameters['routeFilterPrefixes'] = [{'cidr': c} for c in cidr]
    if not public:
        if virtual_gateway_id:
            parameters['virtualGatewayId'] = virtual_gateway_id
        elif direct_connect_gateway_id:
            parameters['directConnectGatewayId'] = direct_connect_gateway_id
    return parameters