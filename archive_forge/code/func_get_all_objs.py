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
def get_all_objs(content, vimtype, folder=None, recurse=True):
    if not folder:
        folder = content.rootFolder
    obj = {}
    container = content.viewManager.CreateContainerView(folder, vimtype, recurse)
    for managed_object_ref in container.view:
        try:
            obj.update({managed_object_ref: managed_object_ref.name})
        except vmodl.fault.ManagedObjectNotFound:
            pass
    return obj