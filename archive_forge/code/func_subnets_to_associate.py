from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def subnets_to_associate(nacl, client, module):
    params = list(module.params.get('subnets'))
    if not params:
        return []
    all_found = []
    if any((x.startswith('subnet-') for x in params)):
        try:
            subnets = _describe_subnets(client, Filters=[{'Name': 'subnet-id', 'Values': params}])
            all_found.extend(subnets.get('Subnets', []))
        except botocore.exceptions.ClientError as e:
            module.fail_json_aws(e)
    if len(params) != len(all_found):
        try:
            subnets = _describe_subnets(client, Filters=[{'Name': 'tag:Name', 'Values': params}])
            all_found.extend(subnets.get('Subnets', []))
        except botocore.exceptions.ClientError as e:
            module.fail_json_aws(e)
    return list(set((s['SubnetId'] for s in all_found if s.get('SubnetId'))))