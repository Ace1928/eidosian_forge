from __future__ import (absolute_import, division, print_function)
import csv
import datetime
import os
import time
import threading
from abc import ABCMeta, abstractmethod
from functools import partial
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible.parsing.ajson import AnsibleJSONEncoder, json
from ansible.plugins.callback import CallbackBase
def dict_fromkeys(keys, default=None):
    d = {}
    for key in keys:
        d[key] = default() if callable(default) else default
    return d