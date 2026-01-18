from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
@staticmethod
def split_link_args(args):
    """
        Split a list of link arguments into three lists:
        * library search paths
        * library filenames (or paths)
        * other link arguments
        """
    lpaths = []
    libs = []
    other = []
    for arg in args:
        if arg.startswith('/LIBPATH:'):
            lpath = arg[9:]
            if lpath in lpaths:
                lpaths.remove(lpath)
            lpaths.append(lpath)
        elif arg.startswith(('/', '-')):
            other.append(arg)
        elif arg.endswith('.lib') or arg.endswith('.a'):
            if arg not in libs:
                libs.append(arg)
        else:
            other.append(arg)
    return (lpaths, libs, other)