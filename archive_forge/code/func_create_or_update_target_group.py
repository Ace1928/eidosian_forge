import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_target_group(connection, module):
    changed = False
    new_target_group = False
    params = dict()
    target_type = module.params.get('target_type')
    params['Name'] = module.params.get('name')
    params['TargetType'] = target_type
    if target_type != 'lambda':
        params['Protocol'] = module.params.get('protocol').upper()
        if module.params.get('protocol_version') is not None:
            params['ProtocolVersion'] = module.params.get('protocol_version')
        params['Port'] = module.params.get('port')
        params['VpcId'] = module.params.get('vpc_id')
    tags = module.params.get('tags')
    purge_tags = module.params.get('purge_tags')
    health_option_keys = ['health_check_path', 'health_check_protocol', 'health_check_interval', 'health_check_timeout', 'healthy_threshold_count', 'unhealthy_threshold_count', 'successful_response_codes']
    health_options = any((module.params[health_option_key] is not None for health_option_key in health_option_keys))
    if health_options:
        if module.params.get('health_check_protocol') is not None:
            params['HealthCheckProtocol'] = module.params.get('health_check_protocol').upper()
        if module.params.get('health_check_port') is not None:
            params['HealthCheckPort'] = module.params.get('health_check_port')
        if module.params.get('health_check_interval') is not None:
            params['HealthCheckIntervalSeconds'] = module.params.get('health_check_interval')
        if module.params.get('health_check_timeout') is not None:
            params['HealthCheckTimeoutSeconds'] = module.params.get('health_check_timeout')
        if module.params.get('healthy_threshold_count') is not None:
            params['HealthyThresholdCount'] = module.params.get('healthy_threshold_count')
        if module.params.get('unhealthy_threshold_count') is not None:
            params['UnhealthyThresholdCount'] = module.params.get('unhealthy_threshold_count')
        protocol = module.params.get('health_check_protocol')
        if protocol is not None and protocol.upper() in ['HTTP', 'HTTPS']:
            if module.params.get('health_check_path') is not None:
                params['HealthCheckPath'] = module.params.get('health_check_path')
            if module.params.get('successful_response_codes') is not None:
                params['Matcher'] = {}
                code_key = 'HttpCode'
                protocol_version = module.params.get('protocol_version')
                if protocol_version is not None and protocol_version.upper() == 'GRPC':
                    code_key = 'GrpcCode'
                params['Matcher'][code_key] = module.params.get('successful_response_codes')
    target_group = get_target_group(connection, module)
    if target_group:
        diffs = [param for param in ('Port', 'Protocol', 'VpcId') if target_group.get(param) != params.get(param)]
        if diffs:
            module.fail_json(msg=f'Cannot modify {', '.join(diffs)} parameter(s) for a target group')
        health_check_params = dict()
        if health_options:
            if 'HealthCheckProtocol' in params and target_group['HealthCheckProtocol'] != params['HealthCheckProtocol']:
                health_check_params['HealthCheckProtocol'] = params['HealthCheckProtocol']
            if 'HealthCheckPort' in params and target_group['HealthCheckPort'] != params['HealthCheckPort']:
                health_check_params['HealthCheckPort'] = params['HealthCheckPort']
            if 'HealthCheckIntervalSeconds' in params and target_group['HealthCheckIntervalSeconds'] != params['HealthCheckIntervalSeconds']:
                health_check_params['HealthCheckIntervalSeconds'] = params['HealthCheckIntervalSeconds']
            if 'HealthCheckTimeoutSeconds' in params and target_group['HealthCheckTimeoutSeconds'] != params['HealthCheckTimeoutSeconds']:
                health_check_params['HealthCheckTimeoutSeconds'] = params['HealthCheckTimeoutSeconds']
            if 'HealthyThresholdCount' in params and target_group['HealthyThresholdCount'] != params['HealthyThresholdCount']:
                health_check_params['HealthyThresholdCount'] = params['HealthyThresholdCount']
            if 'UnhealthyThresholdCount' in params and target_group['UnhealthyThresholdCount'] != params['UnhealthyThresholdCount']:
                health_check_params['UnhealthyThresholdCount'] = params['UnhealthyThresholdCount']
            if target_group['HealthCheckProtocol'] in ['HTTP', 'HTTPS']:
                if 'HealthCheckPath' in params and target_group['HealthCheckPath'] != params['HealthCheckPath']:
                    health_check_params['HealthCheckPath'] = params['HealthCheckPath']
                if 'Matcher' in params:
                    code_key = 'HttpCode'
                    if target_group.get('ProtocolVersion') == 'GRPC':
                        code_key = 'GrpcCode'
                    current_matcher_list = target_group['Matcher'][code_key].split(',')
                    requested_matcher_list = params['Matcher'][code_key].split(',')
                    if set(current_matcher_list) != set(requested_matcher_list):
                        health_check_params['Matcher'] = {}
                        health_check_params['Matcher'][code_key] = ','.join(requested_matcher_list)
            try:
                if health_check_params:
                    connection.modify_target_group(TargetGroupArn=target_group['TargetGroupArn'], aws_retry=True, **health_check_params)
                    changed = True
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't update target group")
        if module.params.get('modify_targets'):
            try:
                current_targets = connection.describe_target_health(TargetGroupArn=target_group['TargetGroupArn'], aws_retry=True)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't get target group health")
            if module.params.get('targets'):
                if target_type != 'lambda':
                    params['Targets'] = module.params.get('targets')
                    for target in params['Targets']:
                        target['Port'] = int(target.get('Port', module.params.get('port')))
                    current_instance_ids = []
                    for instance in current_targets['TargetHealthDescriptions']:
                        current_instance_ids.append(instance['Target']['Id'])
                    new_instance_ids = []
                    for instance in params['Targets']:
                        new_instance_ids.append(instance['Id'])
                    add_instances = set(new_instance_ids) - set(current_instance_ids)
                    if add_instances:
                        instances_to_add = []
                        for target in params['Targets']:
                            if target['Id'] in add_instances:
                                tmp_item = {'Id': target['Id'], 'Port': target['Port']}
                                if target.get('AvailabilityZone'):
                                    tmp_item['AvailabilityZone'] = target['AvailabilityZone']
                                instances_to_add.append(tmp_item)
                        changed = True
                        try:
                            connection.register_targets(TargetGroupArn=target_group['TargetGroupArn'], Targets=instances_to_add, aws_retry=True)
                        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                            module.fail_json_aws(e, msg="Couldn't register targets")
                        if module.params.get('wait'):
                            status_achieved, registered_instances = wait_for_status(connection, module, target_group['TargetGroupArn'], instances_to_add, 'healthy')
                            if not status_achieved:
                                module.fail_json(msg='Error waiting for target registration to be healthy - please check the AWS console')
                    remove_instances = set(current_instance_ids) - set(new_instance_ids)
                    if remove_instances:
                        instances_to_remove = []
                        for target in current_targets['TargetHealthDescriptions']:
                            if target['Target']['Id'] in remove_instances:
                                instances_to_remove.append({'Id': target['Target']['Id'], 'Port': target['Target']['Port']})
                        changed = True
                        try:
                            connection.deregister_targets(TargetGroupArn=target_group['TargetGroupArn'], Targets=instances_to_remove, aws_retry=True)
                        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                            module.fail_json_aws(e, msg="Couldn't remove targets")
                        if module.params.get('wait'):
                            status_achieved, registered_instances = wait_for_status(connection, module, target_group['TargetGroupArn'], instances_to_remove, 'unused')
                            if not status_achieved:
                                module.fail_json(msg='Error waiting for target deregistration - please check the AWS console')
                else:
                    try:
                        changed = False
                        target = module.params.get('targets')[0]
                        if len(current_targets['TargetHealthDescriptions']) == 0:
                            changed = True
                        else:
                            for item in current_targets['TargetHealthDescriptions']:
                                if target['Id'] != item['Target']['Id']:
                                    changed = True
                                    break
                        if changed:
                            if target.get('Id'):
                                response = connection.register_targets(TargetGroupArn=target_group['TargetGroupArn'], Targets=[{'Id': target['Id']}], aws_retry=True)
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        module.fail_json_aws(e, msg="Couldn't register targets")
            elif target_type != 'lambda':
                current_instances = current_targets['TargetHealthDescriptions']
                if current_instances:
                    instances_to_remove = []
                    for target in current_targets['TargetHealthDescriptions']:
                        instances_to_remove.append({'Id': target['Target']['Id'], 'Port': target['Target']['Port']})
                    changed = True
                    try:
                        connection.deregister_targets(TargetGroupArn=target_group['TargetGroupArn'], Targets=instances_to_remove, aws_retry=True)
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        module.fail_json_aws(e, msg="Couldn't remove targets")
                    if module.params.get('wait'):
                        status_achieved, registered_instances = wait_for_status(connection, module, target_group['TargetGroupArn'], instances_to_remove, 'unused')
                        if not status_achieved:
                            module.fail_json(msg='Error waiting for target deregistration - please check the AWS console')
            else:
                changed = False
                if current_targets['TargetHealthDescriptions']:
                    changed = True
                    target_to_remove = current_targets['TargetHealthDescriptions'][0]['Target']['Id']
                if changed:
                    connection.deregister_targets(TargetGroupArn=target_group['TargetGroupArn'], Targets=[{'Id': target_to_remove}], aws_retry=True)
    else:
        try:
            connection.create_target_group(aws_retry=True, **params)
            changed = True
            new_target_group = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg="Couldn't create target group")
        target_group = get_target_group(connection, module, retry_missing=True)
        if module.params.get('targets'):
            if target_type != 'lambda':
                params['Targets'] = module.params.get('targets')
                try:
                    connection.register_targets(TargetGroupArn=target_group['TargetGroupArn'], Targets=params['Targets'], aws_retry=True)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    module.fail_json_aws(e, msg="Couldn't register targets")
                if module.params.get('wait'):
                    status_achieved, registered_instances = wait_for_status(connection, module, target_group['TargetGroupArn'], params['Targets'], 'healthy')
                    if not status_achieved:
                        module.fail_json(msg='Error waiting for target registration to be healthy - please check the AWS console')
            else:
                try:
                    target = module.params.get('targets')[0]
                    response = connection.register_targets(TargetGroupArn=target_group['TargetGroupArn'], Targets=[{'Id': target['Id']}], aws_retry=True)
                    changed = True
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    module.fail_json_aws(e, msg="Couldn't register targets")
    attributes_update = create_or_update_attributes(connection, module, target_group, new_target_group)
    if attributes_update:
        changed = True
    if tags is not None:
        current_tags = get_target_group_tags(connection, module, target_group['TargetGroupArn'])
        tags_need_modify, tags_to_delete = compare_aws_tags(boto3_tag_list_to_ansible_dict(current_tags), tags, purge_tags)
        if tags_to_delete:
            try:
                connection.remove_tags(ResourceArns=[target_group['TargetGroupArn']], TagKeys=tags_to_delete, aws_retry=True)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't delete tags from target group")
            changed = True
        if tags_need_modify:
            try:
                connection.add_tags(ResourceArns=[target_group['TargetGroupArn']], Tags=ansible_dict_to_boto3_tag_list(tags_need_modify), aws_retry=True)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't add tags to target group")
            changed = True
    target_group = get_target_group(connection, module)
    target_group.update(get_tg_attributes(connection, module, target_group['TargetGroupArn']))
    snaked_tg = camel_dict_to_snake_dict(target_group)
    snaked_tg['tags'] = boto3_tag_list_to_ansible_dict(get_target_group_tags(connection, module, target_group['TargetGroupArn']))
    module.exit_json(changed=changed, **snaked_tg)