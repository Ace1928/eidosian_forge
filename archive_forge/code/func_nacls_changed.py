from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def nacls_changed(nacl, client, module):
    changed = False
    params = dict()
    params['egress'] = module.params.get('egress')
    params['ingress'] = module.params.get('ingress')
    nacl_id = nacl['NetworkAcls'][0]['NetworkAclId']
    nacl = describe_network_acl(client, module)
    entries = nacl['NetworkAcls'][0]['Entries']
    egress = [rule for rule in entries if rule['Egress'] is True and rule['RuleNumber'] < 32767]
    ingress = [rule for rule in entries if rule['Egress'] is False and rule['RuleNumber'] < 32767]
    if rules_changed(egress, params['egress'], True, nacl_id, client, module):
        changed = True
    if rules_changed(ingress, params['ingress'], False, nacl_id, client, module):
        changed = True
    return changed