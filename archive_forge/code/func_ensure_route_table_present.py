import re
from ipaddress import ip_network
from time import sleep
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def ensure_route_table_present(connection, module):
    gateway_id = module.params.get('gateway_id')
    lookup = module.params.get('lookup')
    propagating_vgw_ids = module.params.get('propagating_vgw_ids')
    purge_routes = module.params.get('purge_routes')
    purge_subnets = module.params.get('purge_subnets')
    purge_tags = module.params.get('purge_tags')
    route_table_id = module.params.get('route_table_id')
    subnets = module.params.get('subnets')
    tags = module.params.get('tags')
    vpc_id = module.params.get('vpc_id')
    routes = create_route_spec(connection, module, vpc_id)
    changed = False
    tags_valid = False
    if lookup == 'tag':
        if tags is not None:
            try:
                route_table = get_route_table_by_tags(connection, module, vpc_id, tags)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Error finding route table with lookup 'tag'")
        else:
            route_table = None
    elif lookup == 'id':
        try:
            route_table = get_route_table_by_id(connection, module, route_table_id)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg="Error finding route table with lookup 'id'")
    if route_table is None:
        changed = True
        if not module.check_mode:
            try:
                create_params = {'VpcId': vpc_id}
                if tags:
                    create_params['TagSpecifications'] = boto3_tag_specifications(tags, types='route-table')
                route_table = connection.create_route_table(aws_retry=True, **create_params)['RouteTable']
                get_waiter(connection, 'route_table_exists').wait(RouteTableIds=[route_table['RouteTableId']])
            except botocore.exceptions.WaiterError as e:
                module.fail_json_aws(e, msg='Timeout waiting for route table creation')
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Error creating route table')
        else:
            route_table = {'id': 'rtb-xxxxxxxx', 'route_table_id': 'rtb-xxxxxxxx', 'vpc_id': vpc_id}
            module.exit_json(changed=changed, route_table=route_table)
    if routes is not None:
        result = ensure_routes(connection=connection, module=module, route_table=route_table, route_specs=routes, purge_routes=purge_routes)
        changed = changed or result
    if propagating_vgw_ids is not None:
        result = ensure_propagation(connection=connection, module=module, route_table=route_table, propagating_vgw_ids=propagating_vgw_ids)
        changed = changed or result
    if not tags_valid and tags is not None:
        changed |= ensure_ec2_tags(connection, module, route_table['RouteTableId'], tags=tags, purge_tags=purge_tags, retry_codes=['InvalidRouteTableID.NotFound'])
        route_table['Tags'] = describe_ec2_tags(connection, module, route_table['RouteTableId'])
    if subnets is not None:
        associated_subnets = find_subnets(connection, module, vpc_id, subnets)
        result = ensure_subnet_associations(connection=connection, module=module, route_table=route_table, subnets=associated_subnets, purge_subnets=purge_subnets)
        changed = changed or result
    if gateway_id == 'None' or gateway_id == '':
        gateway_changed = disassociate_gateway(connection=connection, module=module, route_table=route_table)
    elif gateway_id is not None:
        gateway_changed = associate_gateway(connection=connection, module=module, route_table=route_table, gateway_id=gateway_id)
    else:
        gateway_changed = False
    changed = changed or gateway_changed
    if changed:
        sleep(5)
    module.exit_json(changed=changed, route_table=get_route_table_info(connection, module, route_table))