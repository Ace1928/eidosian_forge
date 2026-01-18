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
def get_validated_source_info(b_source_info_path, namespace, name, version):
    source_info_path = to_text(b_source_info_path, errors='surrogate_or_strict')
    if not os.path.isfile(b_source_info_path):
        return None
    try:
        with open(b_source_info_path, mode='rb') as fd:
            metadata = safe_load(fd)
    except OSError as e:
        display.warning(f"Error getting collection source information at '{source_info_path}': {to_text(e, errors='surrogate_or_strict')}")
        return None
    if not isinstance(metadata, MutableMapping):
        display.warning(f"Error getting collection source information at '{source_info_path}': expected a YAML dictionary")
        return None
    schema_errors = _validate_v1_source_info_schema(namespace, name, version, metadata)
    if schema_errors:
        display.warning(f'Ignoring source metadata file at {source_info_path} due to the following errors:')
        display.warning('\n'.join(schema_errors))
        display.warning('Correct the source metadata file by reinstalling the collection.')
        return None
    return metadata