from __future__ import (absolute_import, division, print_function)
import csv
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import StringIO
from ansible.module_utils.common.text.converters import to_native
def csv_to_list(rawcsv):
    reader_raw = csv.DictReader(StringIO(rawcsv))
    reader = [dict(((k, v.strip()) for k, v in row.items())) for row in reader_raw]
    return list(reader)