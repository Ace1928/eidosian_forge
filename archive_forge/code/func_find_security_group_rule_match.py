from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def find_security_group_rule_match(prototype, security_group_rules):
    matches = [r for r in security_group_rules if is_security_group_rule_match(prototype, r)]
    if len(matches) > 1:
        self.fail_json(msg='Found more a single matching security group rule which match the given parameters.')
    elif len(matches) == 1:
        return matches[0]
    else:
        return None