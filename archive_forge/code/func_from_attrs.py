from __future__ import (absolute_import, division, print_function)
import itertools
import operator
import os
from copy import copy as shallowcopy
from functools import cache
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.dataloader import DataLoader
from ansible.playbook.attribute import Attribute, FieldAttribute, ConnectionFieldAttribute, NonInheritableFieldAttribute
from ansible.plugins.loader import module_loader, action_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier, get_unique_id
def from_attrs(self, attrs):
    """
        Loads attributes from a dictionary
        """
    for attr, value in attrs.items():
        if attr in self.fattributes:
            attribute = self.fattributes[attr]
            if attribute.isa == 'class' and isinstance(value, dict):
                obj = attribute.class_type()
                obj.deserialize(value)
                setattr(self, attr, obj)
            else:
                setattr(self, attr, value)
    self._finalized = True
    self._squashed = True