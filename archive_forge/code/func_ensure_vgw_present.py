import time
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_vgw_present(client, module):
    changed = False
    params = dict()
    result = dict()
    params['Name'] = module.params.get('name')
    params['VpcId'] = module.params.get('vpc_id')
    params['Type'] = module.params.get('type')
    params['Tags'] = module.params.get('tags')
    params['VpnGatewayIds'] = module.params.get('vpn_gateway_id')
    if params['VpcId']:
        vpc = find_vpc(client, module)
    existing_vgw = find_vgw(client, module)
    if existing_vgw != []:
        vpn_gateway_id = existing_vgw[0]['VpnGatewayId']
        desired_tags = module.params.get('tags')
        purge_tags = module.params.get('purge_tags')
        if desired_tags is None:
            desired_tags = dict()
            purge_tags = False
        tags = dict(Name=module.params.get('name'))
        tags.update(desired_tags)
        changed = ensure_ec2_tags(client, module, vpn_gateway_id, resource_type='vpn-gateway', tags=tags, purge_tags=purge_tags)
        if params['VpcId']:
            current_vpc_attachments = existing_vgw[0]['VpcAttachments']
            if current_vpc_attachments != [] and current_vpc_attachments[0]['State'] == 'attached':
                if current_vpc_attachments[0]['VpcId'] != params['VpcId'] or current_vpc_attachments[0]['State'] != 'attached':
                    vpc_to_detach = current_vpc_attachments[0]['VpcId']
                    detach_vgw(client, module, vpn_gateway_id, vpc_to_detach)
                    get_waiter(client, 'vpn_gateway_detached').wait(VpnGatewayIds=[vpn_gateway_id])
                    attached_vgw = attach_vgw(client, module, vpn_gateway_id)
                    changed = True
            else:
                attached_vgw = attach_vgw(client, module, vpn_gateway_id)
                changed = True
        else:
            existing_vgw = find_vgw(client, module, [vpn_gateway_id])
            if existing_vgw[0]['VpcAttachments'] != []:
                if existing_vgw[0]['VpcAttachments'][0]['State'] == 'attached':
                    vpc_to_detach = existing_vgw[0]['VpcAttachments'][0]['VpcId']
                    detach_vgw(client, module, vpn_gateway_id, vpc_to_detach)
                    changed = True
    else:
        new_vgw = create_vgw(client, module)
        changed = True
        vpn_gateway_id = new_vgw['VpnGateway']['VpnGatewayId']
        if params['VpcId']:
            attached_vgw = attach_vgw(client, module, vpn_gateway_id)
            changed = True
    vgw = find_vgw(client, module, [vpn_gateway_id])
    result = get_vgw_info(vgw)
    return (changed, result)