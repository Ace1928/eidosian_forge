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
def require_torch_xpu(test_case):
    """
    Decorator marking a test that requires XPU and IPEX.

    These tests are skipped when Intel Extension for PyTorch isn't installed or it does not match current PyTorch
    version.
    """
    return unittest.skipUnless(is_torch_xpu_available(), 'test requires IPEX and an XPU device')(test_case)