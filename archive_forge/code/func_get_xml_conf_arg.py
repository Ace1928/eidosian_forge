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
def get_xml_conf_arg(cfg, path, data='text'):
    """
    :param cfg: The top level configuration lxml Element tree object
    :param path: The relative xpath w.r.t to top level element (cfg)
           to be searched in the xml hierarchy
    :param data: The type of data to be returned for the matched xml node.
        Valid values are text, tag, attrib, with default as text.
    :return: Returns the required type for the matched xml node or else None
    """
    match = cfg.xpath(path)
    if len(match):
        if data == 'tag':
            result = getattr(match[0], 'tag')
        elif data == 'attrib':
            result = getattr(match[0], 'attrib')
        else:
            result = getattr(match[0], 'text')
    else:
        result = None
    return result