from __future__ import annotations
import ast
import datetime
import os
import re
import sys
from io import BytesIO, TextIOWrapper
import yaml
import yaml.reader
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.yaml import SafeLoader
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.loader import AnsibleLoader
def get_module_name_from_filename(filename, collection):
    if collection:
        path = os.path.join(collection, filename)
    else:
        path = os.path.relpath(filename, 'lib')
    name = os.path.splitext(path)[0].replace(os.path.sep, '.')
    return name