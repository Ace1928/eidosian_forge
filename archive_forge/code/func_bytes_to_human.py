from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
def bytes_to_human(size, isbits=False, unit=None):
    base = 'Bytes'
    if isbits:
        base = 'bits'
    suffix = ''
    for suffix, limit in sorted(iteritems(SIZE_RANGES), key=lambda item: -item[1]):
        if unit is None and size >= limit or (unit is not None and unit.upper() == suffix[0]):
            break
    if limit != 1:
        suffix += base[0]
    else:
        suffix = base
    return '%.2f %s' % (size / limit, suffix)