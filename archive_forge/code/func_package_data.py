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
def package_data(self):
    df = self._pkg_data.get('data_files', [])
    if self.has_mit_lic():
        df.append('LICENSE')
        exclude_files.append('LICENSE')
    pd = self._pkg_data.get('package_data', {})
    if df:
        pd[self.full_package_name] = df
    if sys.version_info < (3,):
        for k in pd:
            if isinstance(k, unicode):
                pd[str(k)] = pd.pop(k)
    return pd