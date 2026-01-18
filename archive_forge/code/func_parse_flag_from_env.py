import asyncio
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Union
from unittest import mock
import torch
import accelerate
from ..state import AcceleratorState, PartialState
from ..utils import (
def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = str_to_bool(value)
        except ValueError:
            raise ValueError(f'If set, {key} must be yes or no.')
    return _value