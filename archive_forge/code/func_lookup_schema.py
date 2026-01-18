from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def lookup_schema(self, schema, ignore_not_found_error=False):
    """Look up schema and return its id"""
    if schema is None:
        return schema
    schema_summary = self.query_objs('schemas/list-identity', key='schemas', displayName=schema)
    if not schema_summary and (not ignore_not_found_error):
        self.fail_json(msg="Provided schema '{0}' does not exist.".format(schema))
    elif (not schema_summary or not schema_summary[0].get('id')) and ignore_not_found_error:
        self.module.warn("Provided schema '{0}' does not exist.".format(schema))
        return None
    schema_id = schema_summary[0].get('id')
    if not schema_id:
        self.fail_json(msg="Schema lookup failed for schema '{0}': '{1}'".format(schema, schema_id))
    return schema_id