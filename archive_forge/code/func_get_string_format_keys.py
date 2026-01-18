from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
def get_string_format_keys(fmt_string, old_style=True):
    """Gets a list of required keys from a format string

    Required mostly for parsing base_path urls for required keys, which
    use the old style string formatting.
    """
    if old_style:

        class AccessSaver:

            def __init__(self):
                self.keys = []

            def __getitem__(self, key):
                self.keys.append(key)
        a = AccessSaver()
        fmt_string % a
        return a.keys
    else:
        keys = []
        for t in string.Formatter().parse(fmt_string):
            if t[1] is not None:
                keys.append(t[1])
        return keys