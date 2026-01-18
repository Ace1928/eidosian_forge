import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps
import setuptools
from pkg_resources import (
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen
def gen_setup(self, filename, fragment, tmpdir):
    match = EGG_FRAGMENT.match(fragment)
    dists = match and [d for d in interpret_distro_name(filename, match.group(1), None) if d.version] or []
    if len(dists) == 1:
        basename = os.path.basename(filename)
        if os.path.dirname(filename) != tmpdir:
            dst = os.path.join(tmpdir, basename)
            if not (os.path.exists(dst) and os.path.samefile(filename, dst)):
                shutil.copy2(filename, dst)
                filename = dst
        with open(os.path.join(tmpdir, 'setup.py'), 'w') as file:
            file.write('from setuptools import setup\nsetup(name=%r, version=%r, py_modules=[%r])\n' % (dists[0].project_name, dists[0].version, os.path.splitext(basename)[0]))
        return filename
    elif match:
        raise DistutilsError("Can't unambiguously interpret project/version identifier %r; any dashes in the name or version should be escaped using underscores. %r" % (fragment, dists))
    else:
        raise DistutilsError("Can't process plain .py files without an '#egg=name-version' suffix to enable automatic setup script generation.")