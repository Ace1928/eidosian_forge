from distutils.filelist import FileList as _FileList
from distutils.errors import DistutilsInternalError
from distutils.util import convert_path
from distutils import log
import distutils.errors
import distutils.filelist
import functools
import os
import re
import sys
import time
import collections
from .._importlib import metadata
from .. import _entry_points, _normalization
from . import _requirestxt
from setuptools import Command
from setuptools.command.sdist import sdist
from setuptools.command.sdist import walk_revctrl
from setuptools.command.setopt import edit_config
from setuptools.command import bdist_egg
import setuptools.unicode_utils as unicode_utils
from setuptools.glob import glob
from setuptools.extern import packaging
from ..warnings import SetuptoolsDeprecationWarning
def save_version_info(self, filename):
    """
        Materialize the value of date into the
        build tag. Install build keys in a deterministic order
        to avoid arbitrary reordering on subsequent builds.
        """
    egg_info = collections.OrderedDict()
    egg_info['tag_build'] = self.tags()
    egg_info['tag_date'] = 0
    edit_config(filename, dict(egg_info=egg_info))