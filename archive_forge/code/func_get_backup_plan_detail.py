from ansible_collections.amazon.aws.plugins.module_utils.backup import get_plan_details
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def get_backup_plan_detail(client, module):
    backup_plan_list = []
    backup_plan_names = module.params.get('backup_plan_names')
    for name in backup_plan_names:
        backup_plan_list.extend(get_plan_details(module, client, name))
    module.exit_json(**{'backup_plans': backup_plan_list})