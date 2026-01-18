from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def encode_rules_as_json(rules):
    """
    Converts the given rules into the equivalent JSON representation according to the documentation:
    https://www.consul.io/docs/guides/acl.html#rule-specification.
    :param rules: the rules
    :return: JSON representation of the given rules
    """
    rules_as_json = defaultdict(dict)
    for rule in rules:
        if rule.pattern is not None:
            if rule.pattern in rules_as_json[rule.scope]:
                raise AssertionError()
            rules_as_json[rule.scope][rule.pattern] = {_POLICY_JSON_PROPERTY: rule.policy}
        else:
            if rule.scope in rules_as_json:
                raise AssertionError()
            rules_as_json[rule.scope] = rule.policy
    return rules_as_json