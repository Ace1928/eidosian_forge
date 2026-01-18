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
def return_schema(for_collection, plugin_type='module'):
    if plugin_type == 'module':
        return_types = Any('bool', 'complex', 'dict', 'float', 'int', 'list', 'raw', 'str')
        element_types = Any(None, 'bits', 'bool', 'bytes', 'dict', 'float', 'int', 'json', 'jsonarg', 'list', 'path', 'raw', 'sid', 'str')
    else:
        return_types = Any(None, 'boolean', 'bool', 'integer', 'int', 'float', 'list', 'dict', 'dictionary', 'path', 'str', 'string', 'raw')
        element_types = return_types
    basic_return_option_schema = {Required('description'): doc_string_or_strings, 'returned': doc_string, 'version_added': version(for_collection), 'version_added_collection': collection_name, 'sample': json_value, 'example': json_value, 'elements': element_types, 'choices': Any([object], (object,))}
    if plugin_type == 'module':
        basic_return_option_schema[Required('type')] = return_types
    else:
        basic_return_option_schema['type'] = return_types
    inner_return_option_schema = dict(basic_return_option_schema)
    inner_return_option_schema.update({'contains': Any(None, *list(({str_type: Self} for str_type in string_types)))})
    return_contains_schema = Any(All(Schema(inner_return_option_schema), Schema(return_contains), Schema(partial(version_added, error_code='option-invalid-version-added'))), Schema(type(None)))
    list_dict_return_contains_schema = [{str_type: return_contains_schema} for str_type in string_types]
    return_option_schema = dict(basic_return_option_schema)
    return_option_schema.update({'contains': Any(None, *list_dict_return_contains_schema)})
    if plugin_type == 'module':
        del return_option_schema['returned']
        return_option_schema[Required('returned')] = Any(*string_types)
    return Any(All(Schema({any_string_types: return_option_schema}), Schema({any_string_types: return_contains}), Schema({any_string_types: partial(version_added, error_code='option-invalid-version-added')})), Schema(type(None)))