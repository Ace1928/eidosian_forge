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
def from_human_size(size: str, units: Optional[List[Tuple[str, Any]]]=None) -> int:
    units = units or POW_10_BYTES
    units_dict = {unit.upper(): value for unit, value in units}
    regex = re.compile('(\\d+\\.?\\d*)\\s*({})?'.format('|'.join(units_dict.keys())), re.IGNORECASE)
    match = re.match(regex, size)
    if not match:
        raise ValueError('size must be of the form `10`, `10B` or `10 B`.')
    factor, unit = (float(match.group(1)), units_dict[match.group(2).upper()] if match.group(2) else 1)
    return int(factor * unit)