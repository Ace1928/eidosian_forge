from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def find_dvspg_by_name(dv_switch, portgroup_name):
    portgroup_name = quote_obj_name(portgroup_name)
    portgroups = dv_switch.portgroup
    for pg in portgroups:
        if pg.name == portgroup_name:
            return pg
    return None