import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_vpn_connections(connection, module):
    params = dict()
    params['Filters'] = ansible_dict_to_boto3_filter_list(module.params.get('filters'))
    params['VpnConnectionIds'] = module.params.get('vpn_connection_ids')
    try:
        result = json.loads(json.dumps(connection.describe_vpn_connections(**params), default=date_handler))
    except ValueError as e:
        module.fail_json_aws(e, msg='Cannot validate JSON data')
    except (ClientError, BotoCoreError) as e:
        module.fail_json_aws(e, msg='Could not describe customer gateways')
    snaked_vpn_connections = [camel_dict_to_snake_dict(vpn_connection) for vpn_connection in result['VpnConnections']]
    if snaked_vpn_connections:
        for vpn_connection in snaked_vpn_connections:
            vpn_connection['tags'] = boto3_tag_list_to_ansible_dict(vpn_connection.get('tags', []))
    module.exit_json(changed=False, vpn_connections=snaked_vpn_connections)