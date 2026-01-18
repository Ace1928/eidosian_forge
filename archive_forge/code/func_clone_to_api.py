from __future__ import (absolute_import, division, print_function)
import warnings
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.txt import (
def clone_to_api(self, record):
    """
        Process a record object (DNSRecord) for sending to API.
        Return a modified clone of the record; the original will not be modified.
        """
    record = record.clone()
    self.process_to_api(record)
    return record