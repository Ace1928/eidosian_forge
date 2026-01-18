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
def parse_conf_cmd_arg(cfg, cmd, res1, res2=None, delete_str='no'):
    """
    Parse config based on command

    :param cfg: A text string which is a line of configuration.
    :param cmd: A text string which is the command to be matched
    :param res1: A text string to be returned if the command is present
    :param res2: A text string to be returned if the negate command
                 is present
    :param delete_str: A text string to identify the start of the
                 negate command
    :rtype: A text string
    :returns: A text string if match is found
    """
    match = re.search('\\n\\s+%s(\\n|$)' % cmd, cfg)
    if match:
        return res1
    if res2 is not None:
        match = re.search('\\n\\s+%s %s(\\n|$)' % (delete_str, cmd), cfg)
        if match:
            return res2
    return None