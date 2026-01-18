from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def encode_rule_as_hcl_string(rule):
    """
    Converts the given rule into the equivalent HCL (string) representation.
    :param rule: the rule
    :return: the equivalent HCL (string) representation of the rule
    """
    if rule.pattern is not None:
        return '%s "%s" {\n  %s = "%s"\n}\n' % (rule.scope, rule.pattern, _POLICY_HCL_PROPERTY, rule.policy)
    else:
        return '%s = "%s"\n' % (rule.scope, rule.policy)