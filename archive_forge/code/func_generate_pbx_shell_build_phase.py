from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_shell_build_phase(self, objects_dict: PbxDict) -> None:
    self.generate_test_shell_build_phase(objects_dict)
    self.generate_regen_shell_build_phase(objects_dict)
    self.generate_custom_target_shell_build_phases(objects_dict)
    self.generate_generator_target_shell_build_phases(objects_dict)