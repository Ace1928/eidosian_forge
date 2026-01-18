from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
@staticmethod
def from_fqcr(ref, ref_type):
    """
        Parse a string as a fully-qualified collection reference, raises ValueError if invalid
        :param ref: collection reference to parse (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')
        :param ref_type: the type of the reference, eg 'module', 'role', 'doc_fragment'
        :return: a populated AnsibleCollectionRef object
        """
    if not AnsibleCollectionRef.is_valid_fqcr(ref):
        raise ValueError('{0} is not a valid collection reference'.format(to_native(ref)))
    ref = to_text(ref, errors='strict')
    ref_type = to_text(ref_type, errors='strict')
    ext = ''
    if ref_type == u'playbook' and ref.endswith(PB_EXTENSIONS):
        resource_splitname = ref.rsplit(u'.', 2)
        package_remnant = resource_splitname[0]
        resource = resource_splitname[1]
        ext = '.' + resource_splitname[2]
    else:
        resource_splitname = ref.rsplit(u'.', 1)
        package_remnant = resource_splitname[0]
        resource = resource_splitname[1]
    package_splitname = package_remnant.split(u'.', 2)
    if len(package_splitname) == 3:
        subdirs = package_splitname[2]
    else:
        subdirs = u''
    collection_name = u'.'.join(package_splitname[0:2])
    return AnsibleCollectionRef(collection_name, subdirs, resource + ext, ref_type)