import collections.abc
import configparser
import enum
import getpass
import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from functools import reduce
from typing import (
from urllib.parse import quote, unquote, urlencode, urlparse, urlsplit
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue, Int32Value, StringValue
import wandb
import wandb.env
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import UsageError
from wandb.proto import wandb_settings_pb2
from wandb.sdk.internal.system.env_probe_helpers import is_aws_lambda
from wandb.sdk.lib import filesystem
from wandb.sdk.lib._settings_toposort_generated import SETTINGS_TOPOLOGICALLY_SORTED
from wandb.sdk.wandb_setup import _EarlyLogger
from .lib import apikey
from .lib.gitlib import GitRepo
from .lib.ipython import _get_python_type
from .lib.runid import generate_id
def is_instance_recursive(obj: Any, type_hint: Any) -> bool:
    if type_hint is Any:
        return True
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if origin is None:
        return isinstance(obj, type_hint)
    if origin is Union:
        return any((is_instance_recursive(obj, arg) for arg in args))
    if issubclass(origin, collections.abc.Mapping):
        if not isinstance(obj, collections.abc.Mapping):
            return False
        key_type, value_type = args
        for key, value in obj.items():
            if not is_instance_recursive(key, key_type) or not is_instance_recursive(value, value_type):
                return False
        return True
    if issubclass(origin, collections.abc.Sequence):
        if not isinstance(obj, collections.abc.Sequence) or isinstance(obj, (str, bytes, bytearray)):
            return False
        if len(args) == 1 and args[0] != ...:
            item_type, = args
            for item in obj:
                if not is_instance_recursive(item, item_type):
                    return False
        elif len(args) == 2 and args[-1] == ...:
            item_type = args[0]
            for item in obj:
                if not is_instance_recursive(item, item_type):
                    return False
        elif len(args) == len(obj):
            for item, item_type in zip(obj, args):
                if not is_instance_recursive(item, item_type):
                    return False
        else:
            return False
        return True
    if issubclass(origin, collections.abc.Set):
        if not isinstance(obj, collections.abc.Set):
            return False
        item_type, = args
        for item in obj:
            if not is_instance_recursive(item, item_type):
                return False
        return True
    return False