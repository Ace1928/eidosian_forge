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
def safe_dict_max(key, data):
    """Safely find the maximum for a given key in a list of dict objects.

    This will find the maximum integer value for specific dictionary key
    across a list of dictionaries. The values for the given key MUST be
    integers, or string representations of an integer.

    The dictionary key does not have to be present in all (or any)
    of the elements/dicts within the data set.

    :param string key: The dictionary key to search for the maximum value.
    :param list data: List of dicts to use for the data set.

    :returns: None if the field was not found in any elements, or
        the maximum value for the field otherwise.
    """
    max_value = None
    for d in data:
        if key in d and d[key] is not None:
            try:
                val = int(d[key])
            except ValueError:
                raise exceptions.SDKException('Search for maximum value failed. Value for {key} is not an integer: {value}'.format(key=key, value=d[key]))
            if max_value is None or val > max_value:
                max_value = val
    return max_value