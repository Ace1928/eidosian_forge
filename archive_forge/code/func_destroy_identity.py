import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def destroy_identity(connection, module):
    identity = module.params.get('identity')
    changed = False
    verification_attributes = get_verification_attributes(connection, module, identity)
    if verification_attributes is not None:
        try:
            if not module.check_mode:
                connection.delete_identity(Identity=identity, aws_retry=True)
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Failed to delete identity {identity}')
        changed = True
    module.exit_json(changed=changed, identity=identity)