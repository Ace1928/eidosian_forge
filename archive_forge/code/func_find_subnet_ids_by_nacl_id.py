from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_subnet_ids_by_nacl_id(nacl_id, client, module):
    try:
        results = _describe_network_acls_retry_missing(client, Filters=[{'Name': 'association.network-acl-id', 'Values': [nacl_id]}])
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)
    if results['NetworkAcls']:
        associations = results['NetworkAcls'][0]['Associations']
        return [s['SubnetId'] for s in associations if s['SubnetId']]
    else:
        return []