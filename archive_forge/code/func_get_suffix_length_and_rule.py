from __future__ import absolute_import, division, print_function
import os.path
import re
from ansible_collections.community.dns.plugins.module_utils.names import InvalidDomainName, split_into_labels, normalize_label
def get_suffix_length_and_rule(self, normalized_labels, icann_only=False):
    """
        Given a list of normalized labels, searches for a matching rule.

        Returns the tuple ``(suffix_length, rule)``. The ``rule`` is never ``None``
        except if ``normalized_labels`` is empty, in which case ``(0, None)`` is returned.

        If ``icann_only`` is set to ``True``, only official ICANN rules are used. If
        ``icann_only`` is ``False`` (default), also private rules are used.
        """
    if not normalized_labels:
        return (0, None)
    rules = []
    for rule in self._rules:
        if icann_only and rule.part != 'icann':
            continue
        if rule.matches(normalized_labels):
            rules.append(rule)
    if not rules:
        rules.append(self._generic_rule)
    rule = select_prevailing_rule(rules)
    suffix_length = len(rule.labels)
    if rule.exception_rule:
        suffix_length -= 1
    return (suffix_length, rule)