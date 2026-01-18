from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_nodegroups(client, module):
    changed = False
    params = dict()
    params['nodegroupName'] = module.params['name']
    params['clusterName'] = module.params['cluster_name']
    params['nodeRole'] = module.params['node_role']
    params['subnets'] = module.params['subnets']
    params['tags'] = module.params['tags'] or {}
    if module.params['ami_type'] is not None:
        params['amiType'] = module.params['ami_type']
    if module.params['disk_size'] is not None:
        params['diskSize'] = module.params['disk_size']
    if module.params['instance_types'] is not None:
        params['instanceTypes'] = module.params['instance_types']
    if module.params['launch_template'] is not None:
        params['launchTemplate'] = dict()
        if module.params['launch_template']['id'] is not None:
            params['launchTemplate']['id'] = module.params['launch_template']['id']
        if module.params['launch_template']['version'] is not None:
            params['launchTemplate']['version'] = module.params['launch_template']['version']
        if module.params['launch_template']['name'] is not None:
            params['launchTemplate']['name'] = module.params['launch_template']['name']
    if module.params['release_version'] is not None:
        params['releaseVersion'] = module.params['release_version']
    if module.params['remote_access'] is not None:
        params['remoteAccess'] = dict()
        if module.params['remote_access']['ec2_ssh_key'] is not None:
            params['remoteAccess']['ec2SshKey'] = module.params['remote_access']['ec2_ssh_key']
        if module.params['remote_access']['source_sg'] is not None:
            params['remoteAccess']['sourceSecurityGroups'] = module.params['remote_access']['source_sg']
    if module.params['capacity_type'] is not None:
        params['capacityType'] = module.params['capacity_type'].upper()
    if module.params['labels'] is not None:
        params['labels'] = module.params['labels']
    if module.params['taints'] is not None:
        params['taints'] = module.params['taints']
    if module.params['update_config'] is not None:
        params['updateConfig'] = dict()
        if module.params['update_config']['max_unavailable'] is not None:
            params['updateConfig']['maxUnavailable'] = module.params['update_config']['max_unavailable']
        if module.params['update_config']['max_unavailable_percentage'] is not None:
            params['updateConfig']['maxUnavailablePercentage'] = module.params['update_config']['max_unavailable_percentage']
    if module.params['scaling_config'] is not None:
        params['scalingConfig'] = snake_dict_to_camel_dict(module.params['scaling_config'])
    wait = module.params.get('wait')
    nodegroup = get_nodegroup(client, module, params['nodegroupName'], params['clusterName'])
    if nodegroup:
        update_params = dict()
        update_params['clusterName'] = params['clusterName']
        update_params['nodegroupName'] = params['nodegroupName']
        if 'launchTemplate' in nodegroup:
            if compare_params_launch_template(module, params, nodegroup):
                update_params['launchTemplate'] = params['launchTemplate']
                if not module.check_mode:
                    try:
                        client.update_nodegroup_version(**update_params)
                    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                        module.fail_json_aws(e, msg="Couldn't update nodegroup.")
                changed |= True
        if compare_params(module, params, nodegroup):
            try:
                if 'launchTemplate' in update_params:
                    update_params.pop('launchTemplate')
                update_params['scalingConfig'] = params['scalingConfig']
                update_params['updateConfig'] = params['updateConfig']
                if not module.check_mode:
                    client.update_nodegroup_config(**update_params)
                changed |= True
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                module.fail_json_aws(e, msg="Couldn't update nodegroup.")
        changed |= validate_tags(client, module, nodegroup)
        changed |= validate_labels(client, module, nodegroup, params['labels'])
        if 'taints' in nodegroup:
            changed |= validate_taints(client, module, nodegroup, params['taints'])
        if wait:
            wait_until(client, module, 'nodegroup_active', params['nodegroupName'], params['clusterName'])
        nodegroup = get_nodegroup(client, module, params['nodegroupName'], params['clusterName'])
        module.exit_json(changed=changed, **camel_dict_to_snake_dict(nodegroup))
    if module.check_mode:
        module.exit_json(changed=True)
    try:
        nodegroup = client.create_nodegroup(**params)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f"Couldn't create Nodegroup {params['nodegroupName']}.")
    if wait:
        wait_until(client, module, 'nodegroup_active', params['nodegroupName'], params['clusterName'])
        nodegroup = get_nodegroup(client, module, params['nodegroupName'], params['clusterName'])
    module.exit_json(changed=True, **camel_dict_to_snake_dict(nodegroup))