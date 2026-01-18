from __future__ import (absolute_import, division, print_function)
import base64
import glob
import hashlib
import json
import ntpath
import os.path
import re
import shlex
import sys
import time
import uuid
import yaml
import datetime
from collections.abc import Mapping
from functools import partial
from random import Random, SystemRandom, shuffle
from jinja2.filters import pass_environment
from ansible.errors import AnsibleError, AnsibleFilterError, AnsibleFilterTypeError
from ansible.module_utils.six import string_types, integer_types, reraise, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.yaml import yaml_load, yaml_load_all
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.template import recursive_check_defined
from ansible.utils.display import Display
from ansible.utils.encrypt import do_encrypt, PASSLIB_AVAILABLE
from ansible.utils.hashing import md5s, checksum_s
from ansible.utils.unicode import unicode_wrap
from ansible.utils.unsafe_proxy import _is_unsafe
from ansible.utils.vars import merge_hash
def from_yaml_all(data):
    if isinstance(data, string_types):
        if _is_unsafe(data):
            data = data._strip_unsafe()
        return yaml_load_all(text_type(to_text(data, errors='surrogate_or_strict')))
    return data