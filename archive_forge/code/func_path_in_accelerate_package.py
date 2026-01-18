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
def path_in_accelerate_package(*components: str) -> Path:
    """
    Get a path within the `accelerate` package's directory.

    Args:
        *components: Components of the path to join after the package directory.

    Returns:
        `Path`: The path to the requested file or directory.
    """
    accelerate_package_dir = Path(inspect.getfile(accelerate)).parent
    return accelerate_package_dir.joinpath(*components)