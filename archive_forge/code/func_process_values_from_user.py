from __future__ import (absolute_import, division, print_function)
import warnings
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.txt import (
def process_values_from_user(self, record_type, values):
    """
        Process a list of record values (strings) after receiving from the user.
        """
    return [self.process_value_from_user(record_type, value) for value in values]