from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def encode_rules_as_hcl_string(rules):
    """
    Converts the given rules into the equivalent HCL (string) representation.
    :param rules: the rules
    :return: the equivalent HCL (string) representation of the rules. Will be None if there is no rules (see internal
    note for justification)
    """
    if len(rules) == 0:
        return None
    rules_as_hcl = ''
    for rule in rules:
        rules_as_hcl += encode_rule_as_hcl_string(rule)
    return rules_as_hcl