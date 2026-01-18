from __future__ import (absolute_import, division, print_function)
import os
import typing as t
from collections.abc import MutableMapping, MutableSequence
from functools import partial
from ansible.errors import AnsibleFileNotFound, AnsibleParserError, AnsibleRuntimeError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import string_types, text_type
from ansible.parsing.yaml.objects import AnsibleSequence, AnsibleUnicode
from ansible.plugins.inventory import BaseFileInventoryPlugin
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import AnsibleUnsafeBytes, AnsibleUnsafeText
def convert_yaml_objects_to_native(obj):
    """Older versions of the ``toml`` python library, and tomllib, don't have
    a pluggable way to tell the encoder about custom types, so we need to
    ensure objects that we pass are native types.

    Used with:
      - ``toml<0.10.0`` where ``toml.TomlEncoder`` is missing
      - ``tomli`` or ``tomllib``

    This function recurses an object and ensures we cast any of the types from
    ``ansible.parsing.yaml.objects`` into their native types, effectively cleansing
    the data before we hand it over to the toml library.

    This function doesn't directly check for the types from ``ansible.parsing.yaml.objects``
    but instead checks for the types those objects inherit from, to offer more flexibility.
    """
    if isinstance(obj, dict):
        return dict(((k, convert_yaml_objects_to_native(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return [convert_yaml_objects_to_native(v) for v in obj]
    elif isinstance(obj, text_type):
        return text_type(obj)
    else:
        return obj