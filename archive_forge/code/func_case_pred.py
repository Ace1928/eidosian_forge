from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def case_pred(test):
    if match_tests is None:
        return True
    for name in test.id().split('.'):
        if fnmatch.fnmatchcase(name, match_tests):
            return True
    return False