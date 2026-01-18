from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_nodegroups(client, module):
    name = module.params.get('name')
    clusterName = module.params['cluster_name']
    existing = get_nodegroup(client, module, name, clusterName)
    wait = module.params.get('wait')
    if not existing:
        module.exit_json(changed=False, msg="Nodegroup '{name}' does not exist")
    if existing['status'] == 'DELETING':
        if wait:
            wait_until(client, module, 'nodegroup_deleted', name, clusterName)
            module.exit_json(changed=False, msg="Nodegroup '{name}' deletion complete")
        module.exit_json(changed=False, msg="Nodegroup '{name}' already in DELETING state")
    if module.check_mode:
        module.exit_json(changed=True, msg="Nodegroup '{name}' deletion would be started (check mode)")
    try:
        client.delete_nodegroup(clusterName=clusterName, nodegroupName=name)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f"Couldn't delete Nodegroup '{name}'.")
    if wait:
        wait_until(client, module, 'nodegroup_deleted', name, clusterName)
        module.exit_json(changed=True, msg="Nodegroup '{name}' deletion complete")
    module.exit_json(changed=True, msg="Nodegroup '{name}' deletion started")