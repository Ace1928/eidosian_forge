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
def scan_egg_link(self, path, entry):
    with open(os.path.join(path, entry)) as raw_lines:
        lines = list(filter(None, map(str.strip, raw_lines)))
    if len(lines) != 2:
        return
    egg_path, setup_path = lines
    for dist in find_distributions(os.path.join(path, egg_path)):
        dist.location = os.path.join(path, *lines)
        dist.precedence = SOURCE_DIST
        self.add(dist)