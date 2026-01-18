from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def tags_changed(nacl_id, client, module):
    tags = module.params.get('tags')
    name = module.params.get('name')
    purge_tags = module.params.get('purge_tags')
    if name is None and tags is None:
        return False
    if module.params.get('tags') is None:
        purge_tags = False
    new_tags = dict()
    if module.params.get('name') is not None:
        new_tags['Name'] = module.params.get('name')
    new_tags.update(module.params.get('tags') or {})
    return ensure_ec2_tags(client, module, nacl_id, tags=new_tags, purge_tags=purge_tags, retry_codes=['InvalidNetworkAclID.NotFound'])