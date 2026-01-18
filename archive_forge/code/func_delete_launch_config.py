import traceback
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_launch_config(connection, module):
    try:
        name = module.params.get('name')
        launch_configs = connection.describe_launch_configurations(LaunchConfigurationNames=[name]).get('LaunchConfigurations')
        if launch_configs:
            connection.delete_launch_configuration(LaunchConfigurationName=launch_configs[0].get('LaunchConfigurationName'))
            module.exit_json(changed=True)
        else:
            module.exit_json(changed=False)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to delete launch configuration')