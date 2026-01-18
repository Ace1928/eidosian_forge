from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def generate_subclass(parent_class):
    """Make a class hash-able by generating a subclass with a __hash__ method that returns the sum of all fields within
    the parent class"""
    dict_of_method_in_subclass = {'__init__': parent_class.__init__, '__hash__': generic_hash, '__eq__': generic_eq}
    subclass_name = 'GeneratedSub' + parent_class.__name__
    generated_sub_class = type(subclass_name, (parent_class,), dict_of_method_in_subclass)
    return generated_sub_class