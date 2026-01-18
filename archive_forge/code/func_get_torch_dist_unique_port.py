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
def get_torch_dist_unique_port():
    """
    Returns a port number that can be fed to `torch.distributed.launch`'s `--master_port` argument.

    Under `pytest-xdist` it adds a delta number based on a worker id so that concurrent tests don't try to use the same
    port at once.
    """
    port = 29500
    uniq_delta = pytest_xdist_worker_id()
    return port + uniq_delta