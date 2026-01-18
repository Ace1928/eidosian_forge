from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
def ensure_dns_record(self, record, type, ttl, values, domain):
    if record == '':
        record = '@'
    records = self.get_records(record, type, domain)
    if records:
        cur_record = records[0]
        do_update = False
        if ttl is not None and cur_record['rrset_ttl'] != ttl:
            do_update = True
        if values is not None and set(cur_record['rrset_values']) != set(values):
            do_update = True
        if do_update:
            if self.module.check_mode:
                result = dict(rrset_type=type, rrset_name=record, rrset_values=values, rrset_ttl=ttl)
            else:
                self.update_record(record, type, values, ttl, domain)
                records = self.get_records(record, type, domain)
                result = records[0]
            self.changed = True
            return (result, self.changed)
        else:
            return (cur_record, self.changed)
    if self.module.check_mode:
        new_record = dict(rrset_type=type, rrset_name=record, rrset_values=values, rrset_ttl=ttl)
        result = new_record
    else:
        result = self.create_record(record, type, values, ttl, domain)
    self.changed = True
    return (result, self.changed)