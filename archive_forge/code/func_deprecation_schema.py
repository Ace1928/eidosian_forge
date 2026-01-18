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
def deprecation_schema(for_collection):
    main_fields = {Required('why'): doc_string, Required('alternative'): doc_string, Required('removed_from_collection'): collection_name, 'removed': Any(True)}
    date_schema = {Required('removed_at_date'): date()}
    date_schema.update(main_fields)
    if for_collection:
        version_schema = {Required('removed_in'): version(for_collection)}
    else:
        version_schema = {Required('removed_in'): deprecation_versions()}
    version_schema.update(main_fields)
    result = Any(Schema(version_schema, extra=PREVENT_EXTRA), Schema(date_schema, extra=PREVENT_EXTRA))
    if for_collection:
        result = All(result, partial(check_removal_version, version_field='removed_in', collection_name_field='removed_from_collection', error_code='invalid-removal-version'))
    return result