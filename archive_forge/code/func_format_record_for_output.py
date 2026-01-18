from __future__ import (absolute_import, division, print_function)
def format_record_for_output(record, record_name, prefix=None, record_converter=None):
    entry = {'prefix': prefix or '', 'type': record.type, 'ttl': record.ttl, 'value': record.target, 'extra': record.extra}
    if record_converter:
        entry['value'] = record_converter.process_value_to_user(entry['type'], entry['value'])
    if record_name is not None:
        entry['record'] = record_name
    return entry