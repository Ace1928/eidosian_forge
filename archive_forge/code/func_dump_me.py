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
def dump_me(self, depth=0):
    """ this is never called from production code, it is here to be used when debugging as a 'complex print' """
    if depth == 0:
        display.debug('DUMPING OBJECT ------------------------------------------------------')
    display.debug('%s- %s (%s, id=%s)' % (' ' * depth, self.__class__.__name__, self, id(self)))
    if hasattr(self, '_parent') and self._parent:
        self._parent.dump_me(depth + 2)
        dep_chain = self._parent.get_dep_chain()
        if dep_chain:
            for dep in dep_chain:
                dep.dump_me(depth + 2)
    if hasattr(self, '_play') and self._play:
        self._play.dump_me(depth + 2)