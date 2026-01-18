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
def parse_requirement_arg(spec):
    try:
        return Requirement.parse(spec)
    except ValueError as e:
        raise DistutilsError('Not a URL, existing file, or requirement spec: %r' % (spec,)) from e