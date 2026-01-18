from __future__ import unicode_literals
from distutils.command import install as du_install
from distutils import log
import email
import email.errors
import os
import re
import sys
import warnings
import pkg_resources
import setuptools
from setuptools.command import develop
from setuptools.command import easy_install
from setuptools.command import egg_info
from setuptools.command import install
from setuptools.command import install_scripts
from setuptools.command import sdist
from pbr import extra_files
from pbr import git
from pbr import options
import pbr.pbr_json
from pbr import testr_command
from pbr import version
import threading
from %(module_name)s import %(import_target)s
import sys
from %(module_name)s import %(import_target)s
def get_reqs_from_files(requirements_files):
    existing = _any_existing(requirements_files)
    deprecated = [f for f in existing if f in PY_REQUIREMENTS_FILES]
    if deprecated:
        warnings.warn("Support for '-pyN'-suffixed requirements files is removed in pbr 5.0 and these files are now ignored. Use environment markers instead. Conflicting files: %r" % deprecated, DeprecationWarning)
    existing = [f for f in existing if f not in PY_REQUIREMENTS_FILES]
    for requirements_file in existing:
        with open(requirements_file, 'r') as fil:
            return fil.read().split('\n')
    return []