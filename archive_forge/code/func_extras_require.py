from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
@property
def extras_require(self):
    """dict of conditions -> extra packages informaton required for installation
        as of setuptools 33 doing `package ; python_version<=2.7' in install_requires
        still doesn't work

        https://www.python.org/dev/peps/pep-0508/
        https://wheel.readthedocs.io/en/latest/index.html#defining-conditional-dependencies
        https://hynek.me/articles/conditional-python-dependencies/
        """
    ep = self._pkg_data.get('extras_require')
    return ep