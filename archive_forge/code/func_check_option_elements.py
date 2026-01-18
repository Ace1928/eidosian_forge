from __future__ import annotations
import re
from ansible.module_utils.compat.version import StrictVersion
from functools import partial
from urllib.parse import urlparse
from voluptuous import ALLOW_EXTRA, PREVENT_EXTRA, All, Any, Invalid, Length, MultipleInvalid, Required, Schema, Self, ValueInvalid, Exclusive
from ansible.constants import DOCUMENTABLE_PLUGINS
from ansible.module_utils.six import string_types
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.quoting import unquote
from ansible.utils.version import SemanticVersion
from ansible.release import __version__
from antsibull_docs_parser import dom
from antsibull_docs_parser.parser import parse, Context
from .utils import parse_isodate
def check_option_elements(v):
    v_type = v.get('type')
    v_elements = v.get('elements')
    if v_type == 'list' and v_elements is None:
        raise _add_ansible_error_code(Invalid('Argument defines type as list but elements is not defined'), error_code='parameter-list-no-elements')
    if v_type != 'list' and v_elements is not None:
        raise _add_ansible_error_code(Invalid('Argument defines parameter elements as %s but it is valid only when value of parameter type is list' % (v_elements,)), error_code='doc-elements-invalid')
    return v