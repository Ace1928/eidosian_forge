from __future__ import annotations
from pathlib import Path
from collections import deque
from contextlib import suppress
from copy import deepcopy
from fnmatch import fnmatch
import argparse
import asyncio
import datetime
import enum
import json
import multiprocessing
import os
import pickle
import platform
import random
import re
import signal
import subprocess
import shlex
import sys
import textwrap
import time
import typing as T
import unicodedata
import xml.etree.ElementTree as et
from . import build
from . import environment
from . import mlog
from .coredata import MesonVersionMismatchException, major_versions_differ
from .coredata import version as coredata_version
from .mesonlib import (MesonException, OptionKey, OrderedSet, RealPathAction,
from .mintro import get_infodir, load_info_file
from .programs import ExternalProgram
from .backend.backends import TestProtocol, TestSerialisation
def emit_progress(self, harness: 'TestHarness') -> None:
    if self.progress_test is None:
        self.flush()
        return
    if len(self.running_tests) == 1:
        count = f'{self.started_tests}/{self.test_count}'
    else:
        count = '{}-{}/{}'.format(self.started_tests - len(self.running_tests) + 1, self.started_tests, self.test_count)
    left = '[{}] {} '.format(count, self.spinner[self.spinner_index])
    self.spinner_index = (self.spinner_index + 1) % len(self.spinner)
    right = '{spaces} {dur:{durlen}}'.format(spaces=' ' * TestResult.maxlen(), dur=int(time.time() - self.progress_test.starttime), durlen=harness.duration_max_len)
    if self.progress_test.timeout:
        right += '/{timeout:{durlen}}'.format(timeout=self.progress_test.timeout, durlen=harness.duration_max_len)
    right += 's'
    details = self.progress_test.get_details()
    if details:
        right += '   ' + details
    line = harness.format(self.progress_test, colorize=True, max_left_width=self.max_left_width, left=left, right=right)
    self.print_progress(line)