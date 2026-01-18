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
def install_extension(self, extension, existing=None, pin=None):
    """Install an extension package into JupyterLab.

        The extension is first validated.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
    extension = _normalize_path(extension)
    extensions = self.info['extensions']
    if extension in self.info['core_extensions']:
        config = self._read_build_config()
        uninstalled = config.get('uninstalled_core_extensions', [])
        if extension in uninstalled:
            self.logger.info('Installing core extension %s' % extension)
            uninstalled.remove(extension)
            config['uninstalled_core_extensions'] = uninstalled
            self._write_build_config(config)
            return True
        return False
    self._ensure_app_dirs()
    with TemporaryDirectory() as tempdir:
        info = self._install_extension(extension, tempdir, pin=pin)
    name = info['name']
    if info['is_dir']:
        config = self._read_build_config()
        local = config.setdefault('local_extensions', {})
        local[name] = info['source']
        self._write_build_config(config)
    if name in extensions:
        other = extensions[name]
        if other['path'] != info['path'] and other['location'] == 'app':
            os.remove(other['path'])
    return True