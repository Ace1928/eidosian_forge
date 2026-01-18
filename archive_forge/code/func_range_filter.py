import contextlib
import fnmatch
import inspect
import re
import uuid
from decorator import decorator
import jmespath
import netifaces
from openstack import _log
from openstack import exceptions
def range_filter(data, key, range_exp):
    """Filter a list by a single range expression.

    :param list data: List of dictionaries to be searched.
    :param string key: Key name to search within the data set.
    :param string range_exp: The expression describing the range of values.

    :returns: A list subset of the original data set.
    :raises: :class:`~openstack.exceptions.SDKException` on invalid range
        expressions.
    """
    filtered = []
    range_exp = str(range_exp).upper()
    if range_exp == 'MIN':
        key_min = safe_dict_min(key, data)
        if key_min is None:
            return []
        for d in data:
            if int(d[key]) == key_min:
                filtered.append(d)
        return filtered
    elif range_exp == 'MAX':
        key_max = safe_dict_max(key, data)
        if key_max is None:
            return []
        for d in data:
            if int(d[key]) == key_max:
                filtered.append(d)
        return filtered
    val_range = parse_range(range_exp)
    if val_range is None:
        raise exceptions.SDKException('Invalid range value: {value}'.format(value=range_exp))
    op = val_range[0]
    if op:
        for d in data:
            d_val = int(d[key])
            if op == '<':
                if d_val < val_range[1]:
                    filtered.append(d)
            elif op == '>':
                if d_val > val_range[1]:
                    filtered.append(d)
            elif op == '<=':
                if d_val <= val_range[1]:
                    filtered.append(d)
            elif op == '>=':
                if d_val >= val_range[1]:
                    filtered.append(d)
        return filtered
    else:
        for d in data:
            if int(d[key]) == val_range[1]:
                filtered.append(d)
        return filtered