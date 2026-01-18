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
def check_option_choices(v):
    v_choices = v.get('choices')
    if not is_iterable(v_choices):
        return v
    if v.get('type') == 'list':
        type_checker, type_name = get_type_checker({'type': v.get('elements')})
    else:
        type_checker, type_name = get_type_checker(v)
    if type_checker is None:
        return v
    for value in v_choices:
        try:
            type_checker(value)
        except Exception as exc:
            raise _add_ansible_error_code(Invalid('Argument defines choices as (%r) but this is incompatible with argument type %s: %s' % (value, type_name, exc)), error_code='doc-choices-incompatible-type')
    return v