from __future__ import (absolute_import, division, print_function)
import warnings
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.txt import (
def process_value_to_user(self, record_type, value):
    """
        Process a record value (string) for sending to the user.
        """
    record = DNSRecord()
    record.type = record_type
    record.target = value
    self.process_to_user(record)
    return record.target