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
def read_many_from_queue(q: 'queue.Queue', max_items: int, queue_timeout: Union[int, float]) -> list:
    try:
        item = q.get(True, queue_timeout)
    except queue.Empty:
        return []
    items = [item]
    for _ in range(max_items):
        try:
            item = q.get_nowait()
        except queue.Empty:
            return items
        items.append(item)
    return items