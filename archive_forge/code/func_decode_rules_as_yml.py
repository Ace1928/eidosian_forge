from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def decode_rules_as_yml(rules_as_yml):
    """
    Converts the given YAML representation of rules into a list of rule domain models.
    :param rules_as_yml: the YAML representation of a collection of rules
    :return: the equivalent domain model to the given rules
    """
    rules = RuleCollection()
    if rules_as_yml:
        for rule_as_yml in rules_as_yml:
            rule_added = False
            for scope in RULE_SCOPES:
                if scope in rule_as_yml:
                    if rule_as_yml[scope] is None:
                        raise ValueError("Rule for '%s' does not have a value associated to the scope" % scope)
                    policy = rule_as_yml[_POLICY_YML_PROPERTY] if _POLICY_YML_PROPERTY in rule_as_yml else rule_as_yml[scope]
                    pattern = rule_as_yml[scope] if _POLICY_YML_PROPERTY in rule_as_yml else None
                    rules.add(Rule(scope, policy, pattern))
                    rule_added = True
                    break
            if not rule_added:
                raise ValueError('A rule requires one of %s and a policy.' % '/'.join(RULE_SCOPES))
    return rules