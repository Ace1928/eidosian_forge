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
@contextmanager
def assert_exception(exception_class: Exception, msg: str=None) -> bool:
    """
    Context manager to assert that the right `Exception` class was raised.

    If `msg` is provided, will check that the message is contained in the raised exception.
    """
    was_ran = False
    try:
        yield
        was_ran = True
    except Exception as e:
        assert isinstance(e, exception_class), f'Expected exception of type {exception_class} but got {type(e)}'
        if msg is not None:
            assert msg in str(e), f"Expected message '{msg}' to be in exception but got '{str(e)}'"
    if was_ran:
        raise AssertionError(f'Expected exception of type {exception_class} but ran without issue.')