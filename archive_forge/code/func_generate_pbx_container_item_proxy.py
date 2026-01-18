from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_container_item_proxy(self, objects_dict: PbxDict) -> None:
    for t in self.build_targets:
        proxy_dict = PbxDict()
        objects_dict.add_item(self.containerproxy_map[t], proxy_dict, 'PBXContainerItemProxy')
        proxy_dict.add_item('isa', 'PBXContainerItemProxy')
        proxy_dict.add_item('containerPortal', self.project_uid, 'Project object')
        proxy_dict.add_item('proxyType', '1')
        proxy_dict.add_item('remoteGlobalIDString', self.native_targets[t])
        proxy_dict.add_item('remoteInfo', '"' + t + '"')