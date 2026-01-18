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
def check_futures(futures: T.Iterable[asyncio.Future]) -> None:
    left = False
    for f in futures:
        if not f.done():
            left = True
        elif not f.cancelled():
            f.result()
    if left:
        raise asyncio.TimeoutError