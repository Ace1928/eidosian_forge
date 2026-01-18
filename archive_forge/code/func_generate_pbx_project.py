from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_project(self, objects_dict: PbxDict) -> None:
    project_dict = PbxDict()
    objects_dict.add_item(self.project_uid, project_dict, 'Project object')
    project_dict.add_item('isa', 'PBXProject')
    attr_dict = PbxDict()
    project_dict.add_item('attributes', attr_dict)
    attr_dict.add_item('BuildIndependentTargetsInParallel', 'YES')
    project_dict.add_item('buildConfigurationList', self.project_conflist, f'Build configuration list for PBXProject "{self.build.project_name}"')
    project_dict.add_item('buildSettings', PbxDict())
    style_arr = PbxArray()
    project_dict.add_item('buildStyles', style_arr)
    for name, idval in self.buildstylemap.items():
        style_arr.add_item(idval, name)
    project_dict.add_item('compatibilityVersion', '"Xcode 3.2"')
    project_dict.add_item('hasScannedForEncodings', 0)
    project_dict.add_item('mainGroup', self.maingroup_id)
    project_dict.add_item('projectDirPath', '"' + self.environment.get_source_dir() + '"')
    project_dict.add_item('projectRoot', '""')
    targets_arr = PbxArray()
    project_dict.add_item('targets', targets_arr)
    targets_arr.add_item(self.all_id, 'ALL_BUILD')
    targets_arr.add_item(self.test_id, 'RUN_TESTS')
    targets_arr.add_item(self.regen_id, 'REGENERATE')
    for t in self.build_targets:
        targets_arr.add_item(self.native_targets[t], t)
    for t in self.custom_targets:
        targets_arr.add_item(self.custom_aggregate_targets[t], t)