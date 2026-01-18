from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def subnets_changed(nacl, client, module):
    changed = False
    vpc_id = module.params.get('vpc_id')
    nacl_id = nacl['NetworkAcls'][0]['NetworkAclId']
    subnets = subnets_to_associate(nacl, client, module)
    if not subnets:
        default_nacl_id = find_default_vpc_nacl(vpc_id, client, module)[0]
        subnets = find_subnet_ids_by_nacl_id(nacl_id, client, module)
        if subnets:
            replace_network_acl_association(default_nacl_id, subnets, client, module)
            changed = True
            return changed
        changed = False
        return changed
    subs_added = subnets_added(nacl_id, subnets, client, module)
    if subs_added:
        replace_network_acl_association(nacl_id, subs_added, client, module)
        changed = True
    subs_removed = subnets_removed(nacl_id, subnets, client, module)
    if subs_removed:
        default_nacl_id = find_default_vpc_nacl(vpc_id, client, module)[0]
        replace_network_acl_association(default_nacl_id, subs_removed, client, module)
        changed = True
    return changed