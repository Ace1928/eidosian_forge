from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_launch_configs(connection, module):
    launch_config_name = module.params.get('name')
    sort = module.params.get('sort')
    sort_order = module.params.get('sort_order')
    sort_start = module.params.get('sort_start')
    sort_end = module.params.get('sort_end')
    try:
        pg = connection.get_paginator('describe_launch_configurations')
        launch_configs = pg.paginate(LaunchConfigurationNames=launch_config_name).build_full_result()
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e, msg='Failed to list launch configs')
    snaked_launch_configs = []
    for launch_config in launch_configs['LaunchConfigurations']:
        snaked_launch_configs.append(camel_dict_to_snake_dict(launch_config))
    for launch_config in snaked_launch_configs:
        if 'CreatedTime' in launch_config:
            launch_config['CreatedTime'] = str(launch_config['CreatedTime'])
    if sort:
        snaked_launch_configs.sort(key=lambda e: e[sort], reverse=sort_order == 'descending')
    if sort and sort_start and sort_end:
        snaked_launch_configs = snaked_launch_configs[sort_start:sort_end]
    elif sort and sort_start:
        snaked_launch_configs = snaked_launch_configs[sort_start:]
    elif sort and sort_end:
        snaked_launch_configs = snaked_launch_configs[:sort_end]
    module.exit_json(launch_configurations=snaked_launch_configs)