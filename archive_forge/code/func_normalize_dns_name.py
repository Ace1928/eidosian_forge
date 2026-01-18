from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.names import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def normalize_dns_name(name):
    if name is None:
        return name
    labels, dummy = split_into_labels(name)
    return join_labels([normalize_label(label) for label in labels])