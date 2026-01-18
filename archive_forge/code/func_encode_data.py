import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def encode_data(self, data, attributes):
    current_row = 0
    num_attributes = len(attributes)
    for row in data:
        new_data = []
        if len(row) > 0 and max(row) >= num_attributes:
            raise BadObject('Instance %d has %d attributes, expected %d' % (current_row, max(row) + 1, num_attributes))
        for col in sorted(row):
            v = row[col]
            if v is None or v == '' or v != v:
                s = '?'
            else:
                s = encode_string(str(v))
            new_data.append('%d %s' % (col, s))
        current_row += 1
        yield ' '.join(['{', ','.join(new_data), '}'])