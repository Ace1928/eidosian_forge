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
def data_files(self):
    df = self._pkg_data.get('data_files', [])
    if self.has_mit_lic():
        df.append('LICENSE')
    if not df:
        return None
    return [('.', df)]