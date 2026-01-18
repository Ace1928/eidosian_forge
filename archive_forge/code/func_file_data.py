import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def file_data(value, yaml_loader=None):
    """
    Method decorator to add to your test methods.

    Should be added to methods of instances of ``unittest.TestCase``.

    ``value`` should be a path relative to the directory of the file
    containing the decorated ``unittest.TestCase``. The file
    should contain JSON encoded data, that can either be a list or a
    dict.

    In case of a list, each value in the list will correspond to one
    test case, and the value will be concatenated to the test method
    name.

    In case of a dict, keys will be used as suffixes to the name of the
    test case, and values will be fed as test data.

    ``yaml_loader`` can be used to customize yaml deserialization.
    The default is ``None``, which results in using the ``yaml.safe_load``
    method.
    """

    def wrapper(func):
        setattr(func, FILE_ATTR, value)
        if yaml_loader:
            setattr(func, YAML_LOADER_ATTR, yaml_loader)
        return func
    return wrapper