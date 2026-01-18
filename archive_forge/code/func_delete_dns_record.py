from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
def delete_dns_record(self, record, type, values, domain):
    if record == '':
        record = '@'
    records = self.get_records(record, type, domain)
    if records:
        cur_record = records[0]
        self.changed = True
        if values is not None and set(cur_record['rrset_values']) != set(values):
            new_values = set(cur_record['rrset_values']) - set(values)
            if new_values:
                self.update_record(record, type, list(new_values), cur_record['rrset_ttl'], domain)
                records = self.get_records(record, type, domain)
                return (records[0], self.changed)
        if not self.module.check_mode:
            self.delete_record(record, type, domain)
    else:
        cur_record = None
    return (None, self.changed)