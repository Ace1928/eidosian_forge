import contextlib
import errno
import importlib
import itertools
import os
import platform
import subprocess
import sys
import time
from argparse import Namespace
from collections import namedtuple
import pytest
from pyqtgraph import Qt
from . import utils
def buildFileList(examples, files=None):
    if files is None:
        files = []
    for key, val in examples.items():
        if isinstance(val, dict):
            buildFileList(val, files)
        elif isinstance(val, Namespace):
            files.append((key, val.filename))
        else:
            files.append((key, val))
    return files