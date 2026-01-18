import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def memoized_property(fn):
    attr_name = '_lazy_once_' + fn.__name__

    @property
    def _memoized_property(self):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        else:
            with _memoized_property_lock:
                if not hasattr(self, attr_name):
                    setattr(self, attr_name, fn(self))
            return getattr(self, attr_name)
    return _memoized_property