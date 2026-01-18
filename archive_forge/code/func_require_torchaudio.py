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
def require_torchaudio(test_case):
    """
    Decorator marking a test that requires torchaudio. These tests are skipped when torchaudio isn't installed.
    """
    return unittest.skipUnless(is_torchaudio_available(), 'test requires torchaudio')(test_case)