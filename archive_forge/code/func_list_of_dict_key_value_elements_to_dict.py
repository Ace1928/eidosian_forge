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
def list_of_dict_key_value_elements_to_dict(mylist, key_name='key', value_name='value'):
    """ takes a list of dicts with each having a 'key' and 'value' keys, and transforms the list into a dictionary,
        effectively as the reverse of dict2items """
    if not is_sequence(mylist):
        raise AnsibleFilterTypeError('items2dict requires a list, got %s instead.' % type(mylist))
    try:
        return dict(((item[key_name], item[value_name]) for item in mylist))
    except KeyError:
        raise AnsibleFilterTypeError("items2dict requires each dictionary in the list to contain the keys '%s' and '%s', got %s instead." % (key_name, value_name, mylist))
    except TypeError:
        raise AnsibleFilterTypeError('items2dict requires a list of dictionaries, got %s instead.' % mylist)