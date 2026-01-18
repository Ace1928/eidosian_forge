from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass
@default_collection.setter
def default_collection(cls, value):
    cls._default_collection = value