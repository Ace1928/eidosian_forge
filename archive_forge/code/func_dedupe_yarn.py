import contextlib
import errno
import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import FrozenSet, Optional
from urllib.error import URLError
from urllib.request import Request, quote, urljoin, urlopen
from jupyter_core.paths import jupyter_config_dir
from jupyter_server.extension.serverextension import GREEN_ENABLED, GREEN_OK, RED_DISABLED, RED_X
from jupyterlab_server.config import (
from jupyterlab_server.process import Process, WatchHelper, list2cmdline, which
from packaging.version import Version
from traitlets import Bool, HasTraits, Instance, List, Unicode, default
from jupyterlab._version import __version__
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.jlpmapp import HERE, YARN_PATH
from jupyterlab.semver import Range, gt, gte, lt, lte, make_semver
def dedupe_yarn(path, logger=None):
    """`yarn-deduplicate` with the `fewer` strategy to minimize total
    packages installed in a given staging directory

    This means a extension (or dependency) _could_ cause a downgrade of an
    version expected at publication time, but core should aggressively set
    pins above, for example, known-bad versions
    """
    had_dupes = ProgressProcess(['node', YARN_PATH, 'dlx', 'yarn-berry-deduplicate', '-s', 'fewerHighest', '--fail'], cwd=path, logger=logger).wait() != 0
    if had_dupes:
        yarn_proc = ProgressProcess(['node', YARN_PATH], cwd=path, logger=logger)
        yarn_proc.wait()