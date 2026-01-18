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
def find_runner(program: str) -> Union[None, list, List[str]]:
    """Return a command that will run program.

    Arguments:
        program: The string name of the program to try to run.

    Returns:
        commandline list of strings to run the program (eg. with subprocess.call()) or None
    """
    if os.path.isfile(program) and (not os.access(program, os.X_OK)):
        try:
            opened = open(program)
        except OSError:
            return None
        first_line = opened.readline().strip()
        if first_line.startswith('#!'):
            return shlex.split(first_line[2:])
        if program.endswith('.py'):
            return [sys.executable]
    return None