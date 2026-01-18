from __future__ import (absolute_import, division, print_function)
import os
from collections.abc import Container, Mapping, Set, Sequence
from types import MappingProxyType
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import binary_type, text_type
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.conditional import Conditional
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.role.metadata import RoleMetadata
from ansible.playbook.taggable import Taggable
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.path import is_subpath
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars
def hash_params(params):
    """
    Construct a data structure of parameters that is hashable.

    This requires changing any mutable data structures into immutable ones.
    We chose a frozenset because role parameters have to be unique.

    .. warning::  this does not handle unhashable scalars.  Two things
        mitigate that limitation:

        1) There shouldn't be any unhashable scalars specified in the yaml
        2) Our only choice would be to return an error anyway.
    """
    if isinstance(params, Container) and (not isinstance(params, (text_type, binary_type))):
        if isinstance(params, Mapping):
            try:
                new_params = frozenset(params.items())
            except TypeError:
                new_params = set()
                for k, v in params.items():
                    new_params.add((k, hash_params(v)))
                new_params = frozenset(new_params)
        elif isinstance(params, (Set, Sequence)):
            try:
                new_params = frozenset(params)
            except TypeError:
                new_params = set()
                for v in params:
                    new_params.add(hash_params(v))
                new_params = frozenset(new_params)
        else:
            new_params = frozenset(params)
        return new_params
    return frozenset((params,))