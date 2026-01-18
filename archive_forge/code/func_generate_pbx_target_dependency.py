from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_target_dependency(self, objects_dict: PbxDict) -> None:
    all_dict = PbxDict()
    objects_dict.add_item(self.build_all_tdep_id, all_dict, 'ALL_BUILD')
    all_dict.add_item('isa', 'PBXTargetDependency')
    all_dict.add_item('target', self.all_id)
    targets = []
    targets.append((self.regen_dependency_id, self.regen_id, 'REGEN', None))
    for t in self.build_targets:
        idval = self.pbx_dep_map[t]
        targets.append((idval, self.native_targets[t], t, self.containerproxy_map[t]))
    for t in self.custom_targets:
        idval = self.pbx_custom_dep_map[t]
        targets.append((idval, self.custom_aggregate_targets[t], t, None))
    sorted_targets = sorted(targets, key=operator.itemgetter(0))
    for t in sorted_targets:
        t_dict = PbxDict()
        objects_dict.add_item(t[0], t_dict, 'PBXTargetDependency')
        t_dict.add_item('isa', 'PBXTargetDependency')
        t_dict.add_item('target', t[1], t[2])
        if t[3] is not None:
            t_dict.add_item('targetProxy', t[3], 'PBXContainerItemProxy')