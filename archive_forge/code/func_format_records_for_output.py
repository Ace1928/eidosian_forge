from __future__ import (absolute_import, division, print_function)
def format_records_for_output(records, record_name, prefix=None, record_converter=None):
    ttls = sorted_ttls(set([record.ttl for record in records]))
    entry = {'prefix': prefix or '', 'type': min([record.type for record in records]) if records else None, 'ttl': ttls[0] if len(ttls) > 0 else None, 'value': [record.target for record in records]}
    if record_converter:
        entry['value'] = record_converter.process_values_to_user(entry['type'], entry['value'])
    if record_name is not None:
        entry['record'] = record_name
    if len(ttls) > 1:
        entry['ttls'] = ttls
    return entry