imported modules that pyinstaller would not find on its own using
import os
import sys
import pkgutil
import logging
from os.path import dirname, join
import importlib
import subprocess
import re
import glob
import kivy
from kivy.factory import Factory
from PyInstaller.depend import bindepend
from os import environ
def get_deps_all():
    """Similar to :func:`get_deps_minimal`, but this returns all the
    kivy modules that can indirectly imported. Which includes all
    the possible kivy providers.

    This can be used to get a list of all the possible providers
    which can then manually be included/excluded by commenting out elements
    in the list instead of passing on all the items. See module description.

    :returns:

        A dict with three keys, ``hiddenimports``, ``excludes``, and
        ``binaries``. Their values are a list of the corresponding modules to
        include/exclude. This can be passed directly to `Analysis`` with
        e.g. ::

            a = Analysis(['..\\kivy\\examples\\demo\\touchtracer\\main.py'],
                        ...
                         **get_deps_all())
    """
    return {'binaries': _find_gst_binaries(), 'hiddenimports': sorted(set(kivy_modules + collect_submodules('kivy.core'))), 'excludes': []}