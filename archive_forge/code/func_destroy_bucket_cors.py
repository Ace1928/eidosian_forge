from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def destroy_bucket_cors(connection, module):
    name = module.params.get('name')
    changed = False
    try:
        cors = connection.delete_bucket_cors(Bucket=name)
        changed = True
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg=f'Unable to delete CORS for bucket {name}')
    module.exit_json(changed=changed)