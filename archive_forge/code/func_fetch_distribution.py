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
def fetch_distribution(self, requirement, tmpdir, force_scan=False, source=False, develop_ok=False, local_index=None):
    """Obtain a distribution suitable for fulfilling `requirement`

        `requirement` must be a ``pkg_resources.Requirement`` instance.
        If necessary, or if the `force_scan` flag is set, the requirement is
        searched for in the (online) package index as well as the locally
        installed packages.  If a distribution matching `requirement` is found,
        the returned distribution's ``location`` is the value you would have
        gotten from calling the ``download()`` method with the matching
        distribution's URL or filename.  If no matching distribution is found,
        ``None`` is returned.

        If the `source` flag is set, only source distributions and source
        checkout links will be considered.  Unless the `develop_ok` flag is
        set, development and system eggs (i.e., those using the ``.egg-info``
        format) will be ignored.
        """
    self.info('Searching for %s', requirement)
    skipped = {}
    dist = None

    def find(req, env=None):
        if env is None:
            env = self
        for dist in env[req.key]:
            if dist.precedence == DEVELOP_DIST and (not develop_ok):
                if dist not in skipped:
                    self.warn('Skipping development or system egg: %s', dist)
                    skipped[dist] = 1
                continue
            test = dist in req and (dist.precedence <= SOURCE_DIST or not source)
            if test:
                loc = self.download(dist.location, tmpdir)
                dist.download_location = loc
                if os.path.exists(dist.download_location):
                    return dist
    if force_scan:
        self.prescan()
        self.find_packages(requirement)
        dist = find(requirement)
    if not dist and local_index is not None:
        dist = find(requirement, local_index)
    if dist is None:
        if self.to_scan is not None:
            self.prescan()
        dist = find(requirement)
    if dist is None and (not force_scan):
        self.find_packages(requirement)
        dist = find(requirement)
    if dist is None:
        self.warn('No local packages or working download links found for %s%s', source and 'a source distribution of ' or '', requirement)
    else:
        self.info('Best match: %s', dist)
        return dist.clone(location=dist.download_location)