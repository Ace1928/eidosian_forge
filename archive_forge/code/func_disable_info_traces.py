import email
import itertools
import functools
import os
import posixpath
import re
import zipfile
import contextlib
from distutils.util import get_platform
import setuptools
from setuptools.extern.packaging.version import Version as parse_version
from setuptools.extern.packaging.tags import sys_tags
from setuptools.extern.packaging.utils import canonicalize_name
from setuptools.command.egg_info import write_requirements, _egg_basename
from setuptools.archive_util import _unpack_zipfile_obj
@contextlib.contextmanager
def disable_info_traces():
    """
    Temporarily disable info traces.
    """
    from distutils import log
    saved = log.set_threshold(log.WARN)
    try:
        yield
    finally:
        log.set_threshold(saved)