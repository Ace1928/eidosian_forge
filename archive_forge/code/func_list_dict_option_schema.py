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
def list_dict_option_schema(for_collection, plugin_type):
    if plugin_type == 'module':
        option_types = Any(None, 'bits', 'bool', 'bytes', 'dict', 'float', 'int', 'json', 'jsonarg', 'list', 'path', 'raw', 'sid', 'str')
        element_types = option_types
    else:
        option_types = Any(None, 'boolean', 'bool', 'integer', 'int', 'float', 'list', 'dict', 'dictionary', 'none', 'path', 'tmp', 'temppath', 'tmppath', 'pathspec', 'pathlist', 'str', 'string', 'raw')
        element_types = Any(None, 'boolean', 'bool', 'integer', 'int', 'float', 'list', 'dict', 'dictionary', 'path', 'str', 'string', 'raw')
    basic_option_schema = {Required('description'): doc_string_or_strings, 'required': bool, 'choices': list, 'aliases': Any(list_string_types), 'version_added': version(for_collection), 'version_added_collection': collection_name, 'default': json_value, 'type': option_types, 'elements': element_types}
    if plugin_type != 'module':
        basic_option_schema['name'] = Any(*string_types)
        deprecated_schema = All(Schema(All({'why': doc_string, 'alternatives': doc_string, Exclusive('removed_at_date', 'vod'): date(), Exclusive('version', 'vod'): version(for_collection), 'collection_name': collection_name}, {Required('why'): Any(*string_types), 'alternatives': Any(*string_types), Required(Any('removed_at_date', 'version')): Any(*string_types), Required('collection_name'): Any(*string_types)}), extra=PREVENT_EXTRA), partial(check_removal_version, version_field='version', collection_name_field='collection_name', error_code='invalid-removal-version'))
        env_schema = All(Schema({Required('name'): Any(*string_types), 'deprecated': deprecated_schema, 'version_added': version(for_collection), 'version_added_collection': collection_name}, extra=PREVENT_EXTRA), partial(version_added, error_code='option-invalid-version-added'))
        ini_schema = All(Schema({Required('key'): Any(*string_types), Required('section'): Any(*string_types), 'deprecated': deprecated_schema, 'version_added': version(for_collection), 'version_added_collection': collection_name}, extra=PREVENT_EXTRA), partial(version_added, error_code='option-invalid-version-added'))
        vars_schema = All(Schema({Required('name'): Any(*string_types), 'deprecated': deprecated_schema, 'version_added': version(for_collection), 'version_added_collection': collection_name}, extra=PREVENT_EXTRA), partial(version_added, error_code='option-invalid-version-added'))
        cli_schema = All(Schema({Required('name'): Any(*string_types), 'option': Any(*string_types), 'deprecated': deprecated_schema, 'version_added': version(for_collection), 'version_added_collection': collection_name}, extra=PREVENT_EXTRA), partial(version_added, error_code='option-invalid-version-added'))
        keyword_schema = All(Schema({Required('name'): Any(*string_types), 'deprecated': deprecated_schema, 'version_added': version(for_collection), 'version_added_collection': collection_name}, extra=PREVENT_EXTRA), partial(version_added, error_code='option-invalid-version-added'))
        basic_option_schema.update({'env': [env_schema], 'ini': [ini_schema], 'vars': [vars_schema], 'cli': [cli_schema], 'keyword': [keyword_schema], 'deprecated': deprecated_schema})
    suboption_schema = dict(basic_option_schema)
    suboption_schema.update({'suboptions': Any(None, *list(({str_type: Self} for str_type in string_types)))})
    suboption_schema = Schema(All(suboption_schema, check_option_elements, check_option_choices, check_option_default), extra=PREVENT_EXTRA)
    list_dict_suboption_schema = [{str_type: suboption_schema} for str_type in string_types]
    option_schema = dict(basic_option_schema)
    option_schema.update({'suboptions': Any(None, *list_dict_suboption_schema)})
    option_schema = Schema(All(option_schema, check_option_elements, check_option_choices, check_option_default), extra=PREVENT_EXTRA)
    option_version_added = Schema(All({'suboptions': Any(None, *[{str_type: Self} for str_type in string_types])}, partial(version_added, error_code='option-invalid-version-added')), extra=ALLOW_EXTRA)
    return [{str_type: All(option_schema, option_version_added)} for str_type in string_types]