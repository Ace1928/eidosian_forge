from __future__ import annotations
import os
import json
import re
import sys
import shutil
import typing as T
from collections import defaultdict
from pathlib import Path
from . import mlog
from . import mesonlib
from .mesonlib import MesonException, RealPathAction, join_args, setup_vsenv
from mesonbuild.environment import detect_ninja
from mesonbuild.coredata import UserArrayOption
from mesonbuild import build
def generate_target_names_ninja(target: ParsedTargetName, builddir: Path, introspect_data: dict) -> T.List[str]:
    intro_target = get_target_from_intro_data(target, builddir, introspect_data)
    if intro_target['type'] in {'alias', 'run'}:
        return [target.name]
    else:
        return [str(Path(out_file).relative_to(builddir.resolve())) for out_file in intro_target['filename']]