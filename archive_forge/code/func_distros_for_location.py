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
def distros_for_location(location, basename, metadata=None):
    """Yield egg or source distribution objects based on basename"""
    if basename.endswith('.egg.zip'):
        basename = basename[:-4]
    if basename.endswith('.egg') and '-' in basename:
        return [Distribution.from_location(location, basename, metadata)]
    if basename.endswith('.whl') and '-' in basename:
        wheel = Wheel(basename)
        if not wheel.is_compatible():
            return []
        return [Distribution(location=location, project_name=wheel.project_name, version=wheel.version, precedence=EGG_DIST + 1)]
    if basename.endswith('.exe'):
        win_base, py_ver, platform = parse_bdist_wininst(basename)
        if win_base is not None:
            return interpret_distro_name(location, win_base, metadata, py_ver, BINARY_DIST, platform)
    for ext in EXTENSIONS:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            return interpret_distro_name(location, basename, metadata)
    return []