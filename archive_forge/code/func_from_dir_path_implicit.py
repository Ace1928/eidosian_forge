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
@classmethod
def from_dir_path_implicit(cls, dir_path):
    """Construct a collection instance based on an arbitrary dir.

        This alternative constructor infers the FQCN based on the parent
        and current directory names. It also sets the version to "*"
        regardless of whether any of known metadata files are present.
        """
    if dir_path.endswith(to_bytes(os.path.sep)):
        dir_path = dir_path.rstrip(to_bytes(os.path.sep))
    u_dir_path = to_text(dir_path, errors='surrogate_or_strict')
    path_list = u_dir_path.split(os.path.sep)
    req_name = '.'.join(path_list[-2:])
    return cls(req_name, '*', dir_path, 'dir', None)