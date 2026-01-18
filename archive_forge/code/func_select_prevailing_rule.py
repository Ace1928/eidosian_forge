from __future__ import absolute_import, division, print_function
import os.path
import re
from ansible_collections.community.dns.plugins.module_utils.names import InvalidDomainName, split_into_labels, normalize_label
def select_prevailing_rule(rules):
    """
    Given a non-empty set of rules matching a domain name, finds the prevailing rule.

    It uses the algorithm specified on https://publicsuffix.org/list/.
    """
    max_length_rule = rules[0]
    max_length = len(max_length_rule.labels)
    for rule in rules:
        if rule.exception_rule:
            return rule
        if len(rule.labels) > max_length:
            max_length = len(rule.labels)
            max_length_rule = rule
    return max_length_rule