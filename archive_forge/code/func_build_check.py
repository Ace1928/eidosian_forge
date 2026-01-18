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
def build_check(self, fast=None):
    """Determine whether JupyterLab should be built.

        Returns a list of messages.
        """
    if fast is None:
        fast = self.skip_full_build_check
    app_dir = self.app_dir
    local = self.info['local_extensions']
    linked = self.info['linked_packages']
    messages = []
    pkg_path = pjoin(app_dir, 'static', 'package.json')
    if not osp.exists(pkg_path):
        return ['No built application']
    static_data = self.info['static_data']
    old_jlab = static_data['jupyterlab']
    old_deps = static_data.get('dependencies', {})
    static_version = old_jlab.get('version', '')
    if not static_version.endswith('-spliced'):
        core_version = old_jlab['version']
        if Version(static_version) != Version(core_version):
            msg = 'Version mismatch: %s (built), %s (current)'
            return [msg % (static_version, core_version)]
    shadowed_exts = self.info['shadowed_exts']
    new_package = self._get_package_template(silent=fast)
    new_jlab = new_package['jupyterlab']
    new_deps = new_package.get('dependencies', {})
    for ext_type in ['extensions', 'mimeExtensions']:
        for ext in new_jlab[ext_type]:
            if ext in shadowed_exts:
                continue
            if ext not in old_jlab[ext_type]:
                messages.append('%s needs to be included in build' % ext)
        for ext in old_jlab[ext_type]:
            if ext in shadowed_exts:
                continue
            if ext not in new_jlab[ext_type]:
                messages.append('%s needs to be removed from build' % ext)
    src_pkg_dir = pjoin(REPO_ROOT, 'packages')
    for pkg, dep in new_deps.items():
        if old_deps.get(pkg, '').startswith(src_pkg_dir):
            continue
        if pkg not in old_deps:
            continue
        if pkg in local or pkg in linked:
            continue
        if old_deps[pkg] != dep:
            msg = '%s changed from %s to %s'
            messages.append(msg % (pkg, old_deps[pkg], new_deps[pkg]))
    for name, source in local.items():
        if fast or name in shadowed_exts:
            continue
        dname = pjoin(app_dir, 'extensions')
        if self._check_local(name, source, dname):
            messages.append('%s content changed' % name)
    for name, item in linked.items():
        if fast or name in shadowed_exts:
            continue
        dname = pjoin(app_dir, 'staging', 'linked_packages')
        if self._check_local(name, item['source'], dname):
            messages.append('%s content changed' % name)
    return messages