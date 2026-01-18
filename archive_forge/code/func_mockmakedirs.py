import os
import shutil
import time
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from ..utils import (
def mockmakedirs(path, exist_ok=False):
    """Raise an exception to mimic permission issues"""
    raise PermissionError('Fake error')