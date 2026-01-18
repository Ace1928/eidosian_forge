import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_compute_environment(module, client):
    """
    Adds a Batch compute environment

    :param module:
    :param client:
    :return:
    """
    changed = False
    params = ('compute_environment_name', 'type', 'service_role')
    api_params = set_api_params(module, params)
    if module.params['compute_environment_state'] is not None:
        api_params['state'] = module.params['compute_environment_state']
    compute_resources_param_list = ('minv_cpus', 'maxv_cpus', 'desiredv_cpus', 'instance_types', 'image_id', 'subnets', 'security_group_ids', 'ec2_key_pair', 'instance_role', 'tags', 'bid_percentage', 'spot_iam_fleet_role')
    compute_resources_params = set_api_params(module, compute_resources_param_list)
    if module.params['compute_resource_type'] is not None:
        compute_resources_params['type'] = module.params['compute_resource_type']
    api_params['computeResources'] = compute_resources_params
    try:
        if not module.check_mode:
            client.create_compute_environment(**api_params)
        changed = True
    except (ClientError, BotoCoreError) as e:
        module.fail_json_aws(e, msg='Error creating compute environment')
    return changed