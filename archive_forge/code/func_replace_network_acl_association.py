from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def replace_network_acl_association(nacl_id, subnets, client, module):
    params = dict()
    params['NetworkAclId'] = nacl_id
    for association in describe_acl_associations(subnets, client, module):
        params['AssociationId'] = association
        try:
            if not module.check_mode:
                _replace_network_acl_association(client, **params)
        except botocore.exceptions.ClientError as e:
            module.fail_json_aws(e)