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
def interpret_distro_name(location, basename, metadata, py_version=None, precedence=SOURCE_DIST, platform=None):
    """Generate the interpretation of a source distro name

    Note: if `location` is a filesystem filename, you should call
    ``pkg_resources.normalize_path()`` on it before passing it to this
    routine!
    """
    parts = basename.split('-')
    if not py_version and any((re.match('py\\d\\.\\d$', p) for p in parts[2:])):
        return
    for p in range(len(parts)):
        if parts[p][:1].isdigit():
            break
    else:
        p = len(parts)
    yield Distribution(location, metadata, '-'.join(parts[:p]), '-'.join(parts[p:]), py_version=py_version, precedence=precedence, platform=platform)