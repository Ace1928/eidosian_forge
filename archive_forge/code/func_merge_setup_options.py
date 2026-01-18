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
def merge_setup_options(self, options: argparse.Namespace, test: TestSerialisation) -> T.Dict[str, str]:
    current = self.get_test_setup(test)
    if not options.gdb:
        options.gdb = current.gdb
    if options.gdb:
        options.verbose = True
    if options.timeout_multiplier is None:
        options.timeout_multiplier = current.timeout_multiplier
    if options.wrapper is None:
        options.wrapper = current.exe_wrapper
    elif current.exe_wrapper:
        sys.exit('Conflict: both test setup and command line specify an exe wrapper.')
    return current.env.get_env(os.environ.copy())