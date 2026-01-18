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
def get_test_setup(self, test: T.Optional[TestSerialisation]) -> build.TestSetup:
    if ':' in self.options.setup:
        if self.options.setup not in self.build_data.test_setups:
            sys.exit(f"Unknown test setup '{self.options.setup}'.")
        return self.build_data.test_setups[self.options.setup]
    else:
        full_name = test.project_name + ':' + self.options.setup
        if full_name not in self.build_data.test_setups:
            sys.exit(f"Test setup '{self.options.setup}' not found from project '{test.project_name}'.")
        return self.build_data.test_setups[full_name]