import abc
import contextlib
import copy
import hashlib
import os
import threading
from oslo_utils import reflection
from oslo_utils import strutils
import requests
from novaclient import exceptions
from novaclient import utils
def convert_into_with_meta(self, item, resp):
    if isinstance(item, str):
        return StrWithMeta(item, resp)
    elif isinstance(item, bytes):
        return BytesWithMeta(item, resp)
    elif isinstance(item, list):
        return ListWithMeta(item, resp)
    elif isinstance(item, tuple):
        return TupleWithMeta(item, resp)
    elif item is None:
        return TupleWithMeta((), resp)
    else:
        return DictWithMeta(item, resp)