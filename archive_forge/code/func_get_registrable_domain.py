from __future__ import absolute_import, division, print_function
import os.path
import re
from ansible_collections.community.dns.plugins.module_utils.names import InvalidDomainName, split_into_labels, normalize_label
def get_registrable_domain(self, domain, keep_unknown_suffix=True, only_if_registerable=True, normalize_result=False, icann_only=False):
    """
        Given a domain name, extracts the registrable domain. This is the public suffix
        including the last label before the suffix.

        If ``keep_unknown_suffix`` is set to ``False``, only suffixes matching explicit
        entries from the PSL are returned. If no suffix can be found, ``''`` is returned.
        If ``keep_unknown_suffix`` is ``True`` (default), the implicit ``*`` rule is used
        if no other rule matches.

        If ``only_if_registerable`` is set to ``False``, the public suffix is returned
        if there is no label before the suffix. If ``only_if_registerable`` is ``True``
        (default), ``''`` is returned in that case.

        If ``normalize_result`` is set to ``True``, the result is re-combined form the
        normalized labels. In that case, the result is lower-case ASCII. If
        ``normalize_result`` is ``False`` (default), the result ``result`` always satisfies
        ``domain.endswith(result)``.

        If ``icann_only`` is set to ``True``, only official ICANN rules are used. If
        ``icann_only`` is ``False`` (default), also private rules are used.
        """
    try:
        labels, tail = split_into_labels(domain)
        normalized_labels = [normalize_label(label) for label in labels]
    except InvalidDomainName:
        return ''
    if normalize_result:
        labels = normalized_labels
    suffix_length, rule = self.get_suffix_length_and_rule(normalized_labels, icann_only=icann_only)
    if rule is None:
        return ''
    if not keep_unknown_suffix and rule is self._generic_rule:
        return ''
    if suffix_length < len(labels):
        suffix_length += 1
    elif only_if_registerable:
        return ''
    return '.'.join(reversed(labels[:suffix_length])) + tail