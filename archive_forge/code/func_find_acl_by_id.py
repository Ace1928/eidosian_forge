from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_acl_by_id(nacl_id, client, module):
    try:
        return _describe_network_acls_retry_missing(client, NetworkAclIds=[nacl_id])
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)