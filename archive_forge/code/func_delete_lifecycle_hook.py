from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_lifecycle_hook(connection, module):
    lch_name = module.params.get('lifecycle_hook_name')
    asg_name = module.params.get('autoscaling_group_name')
    return_object = {}
    return_object['changed'] = False
    try:
        all_hooks = connection.describe_lifecycle_hooks(AutoScalingGroupName=asg_name)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to get Lifecycle Hooks')
    for hook in all_hooks['LifecycleHooks']:
        if hook['LifecycleHookName'] == lch_name:
            lch_params = {'LifecycleHookName': lch_name, 'AutoScalingGroupName': asg_name}
            try:
                if module.check_mode:
                    module.exit_json(changed=True, msg='Would have deleted AutoScalingGroup Lifecycle Hook if not in check_mode.')
                connection.delete_lifecycle_hook(**lch_params)
                return_object['changed'] = True
                return_object['lifecycle_hook_removed'] = {'LifecycleHookName': lch_name, 'AutoScalingGroupName': asg_name}
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Failed to delete LifecycleHook')
        else:
            pass
    module.exit_json(**camel_dict_to_snake_dict(return_object))