import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def json_friendly(obj: Any) -> Union[Tuple[Any, bool], Tuple[Union[None, str, float], bool]]:
    """Convert an object into something that's more becoming of JSON."""
    converted = True
    typename = get_full_typename(obj)
    if is_tf_eager_tensor_typename(typename):
        obj = obj.numpy()
    elif is_tf_tensor_typename(typename):
        try:
            obj = obj.eval()
        except RuntimeError:
            obj = obj.numpy()
    elif is_pytorch_tensor_typename(typename) or is_fastai_tensor_typename(typename):
        try:
            if obj.requires_grad:
                obj = obj.detach()
        except AttributeError:
            pass
        try:
            obj = obj.data
        except RuntimeError:
            pass
        if obj.size():
            obj = obj.cpu().detach().numpy()
        else:
            return (obj.item(), True)
    elif is_jax_tensor_typename(typename):
        obj = get_jax_tensor(obj)
    if is_numpy_array(obj):
        if obj.size == 1:
            obj = obj.flatten()[0]
        elif obj.size <= 32:
            obj = obj.tolist()
    elif np and isinstance(obj, np.generic):
        obj = _numpy_generic_convert(obj)
    elif isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    elif isinstance(obj, (datetime, date)):
        obj = obj.isoformat()
    elif callable(obj):
        obj = f'{obj.__module__}.{obj.__qualname__}' if hasattr(obj, '__qualname__') and hasattr(obj, '__module__') else str(obj)
    elif isinstance(obj, float) and math.isnan(obj):
        obj = None
    elif isinstance(obj, dict) and np:
        obj, converted = _sanitize_numpy_keys(obj)
    elif isinstance(obj, set):
        obj = tuple(obj)
    else:
        converted = False
    if getsizeof(obj) > VALUE_BYTES_LIMIT:
        wandb.termwarn('Serializing object of type {} that is {} bytes'.format(type(obj).__name__, getsizeof(obj)))
    return (obj, converted)