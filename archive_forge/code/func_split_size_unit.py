from __future__ import absolute_import, division, print_function
import re
import os
import math
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def split_size_unit(string, isint=False):
    """Split a string between the size value (int or float) and the unit.
       Support optional space(s) between the numeric value and the unit.
    """
    unit = re.sub('(\\d|\\.)', '', string).strip()
    value = float(re.sub('%s' % unit, '', string).strip())
    if isint and unit in ('B', ''):
        if int(value) != value:
            raise AssertionError('invalid blocksize value: bytes require an integer value')
    if not unit:
        unit = None
        product = int(round(value))
    else:
        if unit not in SIZE_UNITS.keys():
            raise AssertionError('invalid size unit (%s): unit must be one of %s, or none.' % (unit, ', '.join(sorted(SIZE_UNITS, key=SIZE_UNITS.get))))
        product = int(round(value * SIZE_UNITS[unit]))
    return (value, unit, product)