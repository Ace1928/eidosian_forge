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
def scan_egg_links(self, search_path):
    dirs = filter(os.path.isdir, search_path)
    egg_links = ((path, entry) for path in dirs for entry in os.listdir(path) if entry.endswith('.egg-link'))
    list(itertools.starmap(self.scan_egg_link, egg_links))