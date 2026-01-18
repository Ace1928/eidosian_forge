import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch
import urllib3
from transformers import logging as transformers_logging
from .integrations import (
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
import asyncio  # noqa
def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
    """
        Args:
            tmp_dir (`string`, *optional*):
                if `None`:

                   - a unique temporary path will be created
                   - sets `before=True` if `before` is `None`
                   - sets `after=True` if `after` is `None`
                else:

                   - `tmp_dir` will be created
                   - sets `before=True` if `before` is `None`
                   - sets `after=False` if `after` is `None`
            before (`bool`, *optional*):
                If `True` and the `tmp_dir` already exists, make sure to empty it right away if `False` and the
                `tmp_dir` already exists, any existing files will remain there.
            after (`bool`, *optional*):
                If `True`, delete the `tmp_dir` at the end of the test if `False`, leave the `tmp_dir` and its contents
                intact at the end of the test.

        Returns:
            tmp_dir(`string`): either the same value as passed via *tmp_dir* or the path to the auto-selected tmp dir
        """
    if tmp_dir is not None:
        if before is None:
            before = True
        if after is None:
            after = False
        path = Path(tmp_dir).resolve()
        if not tmp_dir.startswith('./'):
            raise ValueError(f'`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`')
        if before is True and path.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    else:
        if before is None:
            before = True
        if after is None:
            after = True
        tmp_dir = tempfile.mkdtemp()
    if after is True:
        self.teardown_tmp_dirs.append(tmp_dir)
    return tmp_dir