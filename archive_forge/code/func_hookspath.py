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
def hookspath():
    """Returns a list with the directory that contains the alternate (not
    the default included with pyinstaller) pyinstaller hook for kivy,
    ``kivy/tools/packaging/pyinstaller_hooks/kivy-hook.py``. It is
    typically used with ``hookspath=hookspath()`` in the spec
    file.

    The default pyinstaller hook returns all the core providers used using
    :func:`get_deps_minimal` to add to its list of hidden imports. This
    alternate hook only included the essential modules and leaves the core
    providers to be included additionally with :func:`get_deps_minimal`
    or :func:`get_deps_all`.
    """
    return [curdir]