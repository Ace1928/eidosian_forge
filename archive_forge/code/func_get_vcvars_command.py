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
def get_vcvars_command(self):
    has_arch_values = 'VSCMD_ARG_TGT_ARCH' in os.environ and 'VSCMD_ARG_HOST_ARCH' in os.environ
    if 'VCINSTALLDIR' in os.environ:
        vs_version = os.environ['VisualStudioVersion'] if 'VisualStudioVersion' in os.environ else None
        relative_path = 'Auxiliary\\Build\\' if vs_version is not None and vs_version >= '15.0' else ''
        script_path = os.environ['VCINSTALLDIR'] + relative_path + 'vcvarsall.bat'
        if os.path.exists(script_path):
            if has_arch_values:
                target_arch = os.environ['VSCMD_ARG_TGT_ARCH']
                host_arch = os.environ['VSCMD_ARG_HOST_ARCH']
            else:
                target_arch = os.environ.get('Platform', 'x86')
                host_arch = target_arch
            arch = host_arch + '_' + target_arch if host_arch != target_arch else target_arch
            return f'"{script_path}" {arch}'
    if 'VS150COMNTOOLS' in os.environ and has_arch_values:
        script_path = os.environ['VS150COMNTOOLS'] + 'VsDevCmd.bat'
        if os.path.exists(script_path):
            return '"%s" -arch=%s -host_arch=%s' % (script_path, os.environ['VSCMD_ARG_TGT_ARCH'], os.environ['VSCMD_ARG_HOST_ARCH'])
    return ''