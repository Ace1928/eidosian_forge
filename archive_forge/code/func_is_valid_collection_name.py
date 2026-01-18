from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
@staticmethod
def is_valid_collection_name(collection_name):
    """
        Validates if the given string is a well-formed collection name (does not look up the collection itself)
        :param collection_name: candidate collection name to validate (a valid name is of the form 'ns.collname')
        :return: True if the collection name passed is well-formed, False otherwise
        """
    collection_name = to_text(collection_name)
    if collection_name.count(u'.') != 1:
        return False
    return all((not iskeyword(ns_or_name) and is_python_identifier(ns_or_name) for ns_or_name in collection_name.split(u'.')))