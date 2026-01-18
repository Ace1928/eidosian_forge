from __future__ import annotations
import hashlib
import hmac
import json
import logging
import os
import pickle
import pprint
import random
import typing as t
import warnings
from binascii import b2a_hex
from datetime import datetime, timezone
from hmac import compare_digest
import zmq.asyncio
from tornado.ioloop import IOLoop
from traitlets import (
from traitlets.config.configurable import Configurable, LoggingConfigurable
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
from zmq.eventloop.zmqstream import ZMQStream
from ._version import protocol_version
from .adapter import adapt
from .jsonutil import extract_dates, json_clean, json_default, squash_dates
def squash_unicode(obj: t.Any) -> t.Any:
    """coerce unicode back to bytestrings."""
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            obj[key] = squash_unicode(obj[key])
            if isinstance(key, str):
                obj[squash_unicode(key)] = obj.pop(key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = squash_unicode(v)
    elif isinstance(obj, str):
        obj = obj.encode('utf8')
    return obj