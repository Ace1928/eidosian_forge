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
def get_nmake_base_meson_command_and_exe_search_paths() -> T.Tuple[str, str]:
    meson_cmd_list = mesonlib.get_meson_command()
    assert len(meson_cmd_list) == 1 or len(meson_cmd_list) == 2
    exe_search_paths = os.path.dirname(meson_cmd_list[0])
    nmake_base_meson_command = os.path.basename(meson_cmd_list[0])
    if len(meson_cmd_list) != 1:
        nmake_base_meson_command += ' "' + meson_cmd_list[1] + '"'
        exe_search_paths += ';' + os.path.dirname(meson_cmd_list[1])
    exe_search_paths += ';C:\\Windows\\system32;C:\\Windows'
    return (nmake_base_meson_command, exe_search_paths)