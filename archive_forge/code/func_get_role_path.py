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
def get_role_path(self):
    return os.path.realpath(self._role_path)