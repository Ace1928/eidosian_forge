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
def get_launch_command(**kwargs) -> list:
    """
    Wraps around `kwargs` to help simplify launching from `subprocess`.

    Example:
    ```python
    # returns ['accelerate', 'launch', '--num_processes=2', '--device_count=2']
    get_launch_command(num_processes=2, device_count=2)
    ```
    """
    command = ['accelerate', 'launch']
    for k, v in kwargs.items():
        if isinstance(v, bool) and v:
            command.append(f'--{k}')
        elif v is not None:
            command.append(f'--{k}={v}')
    return command