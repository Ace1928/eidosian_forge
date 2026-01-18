from __future__ import absolute_import, division, print_function
import ast
import json
import operator
import re
import socket
from copy import deepcopy
from functools import reduce  # forward compatibility for Python 3
from itertools import chain
from ansible.module_utils import basic
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems, string_types
def contains_vars(self, data):
    if isinstance(data, string_types):
        for marker in (self.env.block_start_string, self.env.variable_start_string, self.env.comment_start_string):
            if marker in data:
                return True
    return False