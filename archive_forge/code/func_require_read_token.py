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
from functools import wraps
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
def require_read_token(fn):
    """
    A decorator that loads the HF token for tests that require to load gated models.
    """
    token = os.getenv('HF_HUB_READ_TOKEN')

    @wraps(fn)
    def _inner(*args, **kwargs):
        with patch('huggingface_hub.utils._headers.get_token', return_value=token):
            return fn(*args, **kwargs)
    return _inner