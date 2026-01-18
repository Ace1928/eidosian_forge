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
def is_tool_test(test_case):
    """
    Decorator marking a test as a tool test. If RUN_TOOL_TESTS is set to a falsy value, those tests will be skipped.
    """
    if not _run_tool_tests:
        return unittest.skip('test is a tool test')(test_case)
    else:
        try:
            import pytest
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_tool_test()(test_case)