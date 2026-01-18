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
def auto_project_name(program: Optional[str]) -> str:
    from wandb.sdk.lib.gitlib import GitRepo
    root_dir = GitRepo().root_dir
    if root_dir is None:
        return 'uncategorized'
    root_dir = to_native_slash_path(root_dir)
    repo_name = os.path.basename(root_dir)
    if program is None:
        return str(repo_name)
    if not os.path.isabs(program):
        program = os.path.join(os.curdir, program)
    prog_dir = os.path.dirname(os.path.abspath(program))
    if not prog_dir.startswith(root_dir):
        return str(repo_name)
    project = repo_name
    sub_path = os.path.relpath(prog_dir, root_dir)
    if sub_path != '.':
        project += '-' + sub_path
    return str(project.replace(os.sep, '_'))