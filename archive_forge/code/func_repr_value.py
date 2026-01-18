import functools
import inspect
import sys
import msgpack
import rapidjson
from ruamel import yaml
def repr_value(value):
    """
    Represent a value in human readable form. For long list's this truncates the printed
    representation.

    :param value: The value to represent.
    :return: A string representation.
    :rtype: basestring
    """
    if isinstance(value, list) and len(value) > REPR_LIST_TRUNCATION:
        return '[{},...]'.format(', '.join(map(repr, value[:REPR_LIST_TRUNCATION])))
    else:
        return repr(value)