from __future__ import absolute_import, division, print_function
from ansible_collections.community.dns.plugins.plugin_utils.public_suffix import PUBLIC_SUFFIX_LIST
def remove_registrable_domain(dns_name, keep_trailing_period=False, keep_unknown_suffix=True, only_if_registerable=True, icann_only=False):
    """Given DNS name, returns the part before the registrable_domain."""
    suffix = PUBLIC_SUFFIX_LIST.get_registrable_domain(dns_name, keep_unknown_suffix=keep_unknown_suffix, only_if_registerable=only_if_registerable, normalize_result=False, icann_only=icann_only)
    return _remove_suffix(dns_name, suffix, keep_trailing_period)