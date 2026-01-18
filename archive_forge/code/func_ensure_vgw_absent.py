import time
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_vgw_absent(client, module):
    changed = False
    params = dict()
    result = dict()
    params['Name'] = module.params.get('name')
    params['VpcId'] = module.params.get('vpc_id')
    params['Type'] = module.params.get('type')
    params['Tags'] = module.params.get('tags')
    params['VpnGatewayIds'] = module.params.get('vpn_gateway_id')
    if params['VpnGatewayIds']:
        existing_vgw_with_id = find_vgw(client, module, [params['VpnGatewayIds']])
        if existing_vgw_with_id != [] and existing_vgw_with_id[0]['State'] != 'deleted':
            existing_vgw = existing_vgw_with_id
            if existing_vgw[0]['VpcAttachments'] != [] and existing_vgw[0]['VpcAttachments'][0]['State'] == 'attached':
                if params['VpcId']:
                    if params['VpcId'] != existing_vgw[0]['VpcAttachments'][0]['VpcId']:
                        module.fail_json(msg='The vpc-id provided does not match the vpc-id currently attached - please check the AWS console')
                    else:
                        detach_vgw(client, module, params['VpnGatewayIds'], params['VpcId'])
                        deleted_vgw = delete_vgw(client, module, params['VpnGatewayIds'])
                        changed = True
                else:
                    vpc_to_detach = existing_vgw[0]['VpcAttachments'][0]['VpcId']
                    detach_vgw(client, module, params['VpnGatewayIds'], vpc_to_detach)
                    deleted_vgw = delete_vgw(client, module, params['VpnGatewayIds'])
                    changed = True
            else:
                deleted_vgw = delete_vgw(client, module, params['VpnGatewayIds'])
                changed = True
        else:
            changed = False
            deleted_vgw = 'Nothing to do'
    else:
        if not module.params.get('name') or not module.params.get('type'):
            module.fail_json(msg="A name and type is required when no vgw-id and a status of 'absent' is supplied")
        existing_vgw = find_vgw(client, module)
        if existing_vgw != [] and existing_vgw[0]['State'] != 'deleted':
            vpn_gateway_id = existing_vgw[0]['VpnGatewayId']
            if existing_vgw[0]['VpcAttachments'] != [] and existing_vgw[0]['VpcAttachments'][0]['State'] == 'attached':
                if params['VpcId']:
                    if params['VpcId'] != existing_vgw[0]['VpcAttachments'][0]['VpcId']:
                        module.fail_json(msg='The vpc-id provided does not match the vpc-id currently attached - please check the AWS console')
                    else:
                        detach_vgw(client, module, vpn_gateway_id, params['VpcId'])
                        deleted_vgw = delete_vgw(client, module, vpn_gateway_id)
                        changed = True
                else:
                    vpc_to_detach = existing_vgw[0]['VpcAttachments'][0]['VpcId']
                    detach_vgw(client, module, vpn_gateway_id, vpc_to_detach)
                    changed = True
                    deleted_vgw = delete_vgw(client, module, vpn_gateway_id)
            else:
                deleted_vgw = delete_vgw(client, module, vpn_gateway_id)
                changed = True
        else:
            changed = False
            deleted_vgw = None
    result = deleted_vgw
    return (changed, result)