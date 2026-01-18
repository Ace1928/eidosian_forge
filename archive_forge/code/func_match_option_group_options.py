from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def match_option_group_options(client, module):
    requires_update = False
    new_options = module.params.get('options')
    current_option = get_option_group(client, module)
    if current_option['options'] == [] and new_options:
        requires_update = True
    else:
        for option in current_option['options']:
            for setting_name in new_options:
                if setting_name['option_name'] == option['option_name']:
                    if any((name in option.keys() - ['option_settings', 'vpc_security_group_memberships'] and setting_name[name] != option[name] for name in setting_name)):
                        requires_update = True
                    if any((name in option and name == 'vpc_security_group_memberships' for name in setting_name)):
                        current_sg = set((sg['vpc_security_group_id'] for sg in option['vpc_security_group_memberships']))
                        new_sg = set(setting_name['vpc_security_group_memberships'])
                        if current_sg != new_sg:
                            requires_update = True
                    if any((new_option_setting['name'] == current_option_setting['name'] and new_option_setting['value'] != current_option_setting['value'] for new_option_setting in setting_name['option_settings'] for current_option_setting in option['option_settings'])):
                        requires_update = True
                else:
                    requires_update = True
    return requires_update