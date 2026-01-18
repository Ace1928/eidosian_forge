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
def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires the bitsandbytes library. Will be skipped when the library or its hard dependency torch is not installed.
    """
    if is_bitsandbytes_available() and is_torch_available():
        try:
            import pytest
            return pytest.mark.bitsandbytes(test_case)
        except ImportError:
            return test_case
    else:
        return unittest.skip('test requires bitsandbytes and torch')(test_case)