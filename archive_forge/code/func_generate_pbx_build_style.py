from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_build_style(self, objects_dict: PbxDict) -> None:
    for name, idval in self.buildstylemap.items():
        styledict = PbxDict()
        objects_dict.add_item(idval, styledict, name)
        styledict.add_item('isa', 'PBXBuildStyle')
        settings_dict = PbxDict()
        styledict.add_item('buildSettings', settings_dict)
        settings_dict.add_item('COPY_PHASE_STRIP', 'NO')
        styledict.add_item('name', f'"{name}"')