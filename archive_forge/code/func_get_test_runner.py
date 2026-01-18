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
def get_test_runner(self, test: TestSerialisation) -> SingleTestRunner:
    name = self.get_pretty_suite(test)
    options = deepcopy(self.options)
    if self.options.setup:
        env = self.merge_setup_options(options, test)
    else:
        env = os.environ.copy()
    test_env = test.env.get_env(env)
    env.update(test_env)
    if test.is_cross_built and test.needs_exe_wrapper and test.exe_wrapper and test.exe_wrapper.found():
        env['MESON_EXE_WRAPPER'] = join_args(test.exe_wrapper.get_command())
    return SingleTestRunner(test, env, name, options)