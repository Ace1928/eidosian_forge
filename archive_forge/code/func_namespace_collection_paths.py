from __future__ import (absolute_import, division, print_function)
import os
import typing as t
from collections import namedtuple
from collections.abc import MutableSequence, MutableMapping
from glob import iglob
from urllib.parse import urlparse
from yaml import safe_load
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection import HAS_PACKAGING, PkgReq
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
@property
def namespace_collection_paths(self):
    return [to_native(path) for path in _find_collections_in_subdirs(self.src)]