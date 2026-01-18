from __future__ import (absolute_import, division, print_function)
import warnings
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.txt import (
def process_from_api(self, record):
    """
        Process a record object (DNSRecord) after receiving from API.
        Modifies the record in-place.
        """
    try:
        record.target = to_text(record.target)
        if record.type == 'TXT':
            self._handle_txt_api(False, record)
        return record
    except DNSConversionError as e:
        raise_from(DNSConversionError(u'While processing record from API: {0}'.format(e.error_message)), e)