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
def generate_projects(self, vslite_ctx: dict=None) -> T.List[Project]:
    startup_project = self.environment.coredata.options[OptionKey('backend_startup_project')].value
    projlist: T.List[Project] = []
    startup_idx = 0
    for i, (name, target) in enumerate(self.build.targets.items()):
        if startup_project and startup_project == target.get_basename():
            startup_idx = i
        outdir = Path(self.environment.get_build_dir(), self.get_target_dir(target))
        outdir.mkdir(exist_ok=True, parents=True)
        fname = name + '.vcxproj'
        target_dir = PurePath(self.get_target_dir(target))
        relname = target_dir / fname
        projfile_path = outdir / fname
        proj_uuid = self.environment.coredata.target_guids[name]
        generated = self.gen_vcxproj(target, str(projfile_path), proj_uuid, vslite_ctx)
        if generated:
            projlist.append((name, relname, proj_uuid, target.for_machine))
    if startup_idx:
        projlist.insert(0, projlist.pop(startup_idx))
    return projlist