from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway, payload_from_object
from ansible.module_utils.basic import AnsibleModule
def get_sgr_from_api(security_group_rules, security_group_rule):
    """ Check if a security_group_rule specs are present in security_group_rules
        Return None if no rules match the specs
        Return the rule if found
    """
    for sgr in security_group_rules:
        if sgr['ip_range'] == security_group_rule['ip_range'] and sgr['dest_port_from'] == security_group_rule['dest_port_from'] and (sgr['direction'] == security_group_rule['direction']) and (sgr['action'] == security_group_rule['action']) and (sgr['protocol'] == security_group_rule['protocol']):
            return sgr
    return None