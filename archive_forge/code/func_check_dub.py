from __future__ import annotations
import json
import os
from . import ExtensionModule, ModuleInfo
from .. import mlog
from ..dependencies import Dependency
from ..dependencies.dub import DubDependency
from ..interpreterbase import typed_pos_args
from ..mesonlib import Popen_safe, MesonException, listify
def check_dub(self, state):
    dubbin = state.find_program('dub', silent=True)
    if dubbin.found():
        try:
            p, out = Popen_safe(dubbin.get_command() + ['--version'])[0:2]
            if p.returncode != 0:
                mlog.warning("Found dub {!r} but couldn't run it".format(' '.join(dubbin.get_command())))
                dubbin = False
        except (FileNotFoundError, PermissionError):
            dubbin = False
    else:
        dubbin = False
    if dubbin:
        mlog.log('Found DUB:', mlog.bold(dubbin.get_path()), '(%s)' % out.strip())
    else:
        mlog.log('Found DUB:', mlog.red('NO'))
    return dubbin