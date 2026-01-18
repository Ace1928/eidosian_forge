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
def process_test_result(self, result: TestRun) -> None:
    if result.res is TestResult.TIMEOUT:
        self.timeout_count += 1
    elif result.res is TestResult.SKIP:
        self.skip_count += 1
    elif result.res is TestResult.OK:
        self.success_count += 1
    elif result.res in {TestResult.FAIL, TestResult.ERROR, TestResult.INTERRUPT}:
        self.fail_count += 1
    elif result.res is TestResult.EXPECTEDFAIL:
        self.expectedfail_count += 1
    elif result.res is TestResult.UNEXPECTEDPASS:
        self.unexpectedpass_count += 1
    else:
        sys.exit(f'Unknown test result encountered: {result.res}')
    if result.res.is_bad():
        self.collected_failures.append(result)
    for l in self.loggers:
        l.log(self, result)