from __future__ import absolute_import, division, print_function
from ansible_collections.community.dns.plugins.plugin_utils.public_suffix import PUBLIC_SUFFIX_LIST
def get_public_suffix(dns_name, keep_leading_period=True, keep_unknown_suffix=True, normalize_result=False, icann_only=False):
    """Given DNS name, returns the public suffix."""
    suffix = PUBLIC_SUFFIX_LIST.get_suffix(dns_name, keep_unknown_suffix=keep_unknown_suffix, normalize_result=normalize_result, icann_only=icann_only)
    if suffix and len(suffix) < len(dns_name) and keep_leading_period:
        suffix = '.' + suffix
    return suffix