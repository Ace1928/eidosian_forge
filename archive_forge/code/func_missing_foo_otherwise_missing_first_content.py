from ..bisect_multi import bisect_multi_bytes
from . import TestCase
def missing_foo_otherwise_missing_first_content(location_keys):
    calls.append(location_keys)
    result = []
    for location_key in location_keys:
        if location_key[1] == 'foo' or location_key[0] == 0:
            result.append((location_key, False))
        else:
            result.append((location_key, -1))
    return result