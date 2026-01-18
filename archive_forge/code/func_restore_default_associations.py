from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def restore_default_associations(assoc_ids, default_nacl_id, client, module):
    if assoc_ids:
        params = dict()
        params['NetworkAclId'] = default_nacl_id[0]
        for assoc_id in assoc_ids:
            params['AssociationId'] = assoc_id
            restore_default_acl_association(params, client, module)
        return True