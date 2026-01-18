import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_customer_gateways(connection, module):
    params = dict()
    params['Filters'] = ansible_dict_to_boto3_filter_list(module.params.get('filters'))
    params['CustomerGatewayIds'] = module.params.get('customer_gateway_ids')
    try:
        result = json.loads(json.dumps(connection.describe_customer_gateways(**params), default=date_handler))
    except (ClientError, BotoCoreError) as e:
        module.fail_json_aws(e, msg='Could not describe customer gateways')
    snaked_customer_gateways = [camel_dict_to_snake_dict(gateway) for gateway in result['CustomerGateways']]
    if snaked_customer_gateways:
        for customer_gateway in snaked_customer_gateways:
            customer_gateway['tags'] = boto3_tag_list_to_ansible_dict(customer_gateway.get('tags', []))
            customer_gateway_name = customer_gateway['tags'].get('Name')
            if customer_gateway_name:
                customer_gateway['customer_gateway_name'] = customer_gateway_name
    module.exit_json(changed=False, customer_gateways=snaked_customer_gateways)