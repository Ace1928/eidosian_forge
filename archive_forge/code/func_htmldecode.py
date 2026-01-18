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
def htmldecode(text):
    """
    Decode HTML entities in the given text.

    >>> htmldecode(
    ...     'https://../package_name-0.1.2.tar.gz'
    ...     '?tokena=A&amp;tokenb=B">package_name-0.1.2.tar.gz')
    'https://../package_name-0.1.2.tar.gz?tokena=A&tokenb=B">package_name-0.1.2.tar.gz'
    """
    return entity_sub(decode_entity, text)