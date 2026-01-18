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
def autodetect_vs_version(build: T.Optional[build.Build], interpreter: T.Optional[Interpreter]) -> backends.Backend:
    vs_version = os.getenv('VisualStudioVersion', None)
    vs_install_dir = os.getenv('VSINSTALLDIR', None)
    if not vs_install_dir:
        raise MesonException('Could not detect Visual Studio: Environment variable VSINSTALLDIR is not set!\nAre you running meson from the Visual Studio Developer Command Prompt?')
    if vs_version == '11.0' or 'Visual Studio 11' in vs_install_dir:
        from mesonbuild.backend.vs2012backend import Vs2012Backend
        return Vs2012Backend(build, interpreter)
    if vs_version == '12.0' or 'Visual Studio 12' in vs_install_dir:
        from mesonbuild.backend.vs2013backend import Vs2013Backend
        return Vs2013Backend(build, interpreter)
    if vs_version == '14.0' or 'Visual Studio 14' in vs_install_dir:
        from mesonbuild.backend.vs2015backend import Vs2015Backend
        return Vs2015Backend(build, interpreter)
    if vs_version == '15.0' or 'Visual Studio 17' in vs_install_dir or 'Visual Studio\\2017' in vs_install_dir:
        from mesonbuild.backend.vs2017backend import Vs2017Backend
        return Vs2017Backend(build, interpreter)
    if vs_version == '16.0' or 'Visual Studio 19' in vs_install_dir or 'Visual Studio\\2019' in vs_install_dir:
        from mesonbuild.backend.vs2019backend import Vs2019Backend
        return Vs2019Backend(build, interpreter)
    if vs_version == '17.0' or 'Visual Studio 22' in vs_install_dir or 'Visual Studio\\2022' in vs_install_dir:
        from mesonbuild.backend.vs2022backend import Vs2022Backend
        return Vs2022Backend(build, interpreter)
    if 'Visual Studio 10.0' in vs_install_dir:
        return Vs2010Backend(build, interpreter)
    raise MesonException('Could not detect Visual Studio using VisualStudioVersion: {!r} or VSINSTALLDIR: {!r}!\nPlease specify the exact backend to use.'.format(vs_version, vs_install_dir))