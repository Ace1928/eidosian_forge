from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def construct_acl_entries(nacl, client, module):
    for entry in module.params.get('ingress'):
        params = process_rule_entry(entry, Egress=False)
        params['NetworkAclId'] = nacl['NetworkAcl']['NetworkAclId']
        create_network_acl_entry(params, client, module)
    for rule in module.params.get('egress'):
        params = process_rule_entry(rule, Egress=True)
        params['NetworkAclId'] = nacl['NetworkAcl']['NetworkAclId']
        create_network_acl_entry(params, client, module)