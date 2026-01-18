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
def get_target_from_intro_data(target: ParsedTargetName, builddir: Path, introspect_data: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
    if target.name not in introspect_data and target.base_name not in introspect_data:
        raise MesonException(f"Can't invoke target `{target.full_name}`: target not found")
    intro_targets = introspect_data[target.name]
    if not intro_targets:
        intro_targets = introspect_data[target.base_name]
    found_targets: T.List[T.Dict[str, T.Any]] = []
    resolved_bdir = builddir.resolve()
    if not target.type and (not target.path) and (not target.suffix):
        found_targets = intro_targets
    else:
        for intro_target in intro_targets:
            intro_target_name = intro_target['name']
            split = intro_target['id'].rsplit('@', 1)
            if len(split) > 1:
                split = split[0].split('@@', 1)
                if len(split) > 1:
                    intro_target_name = split[1]
                else:
                    intro_target_name = split[0]
            if target.type and target.type != intro_target['type'].replace(' ', '_') or target.name != intro_target_name or (target.path and intro_target['filename'] != 'no_name' and (Path(target.path) != Path(intro_target['filename'][0]).relative_to(resolved_bdir).parent)):
                continue
            found_targets += [intro_target]
    if not found_targets:
        raise MesonException(f"Can't invoke target `{target.full_name}`: target not found")
    elif len(found_targets) > 1:
        suggestions: T.List[str] = []
        for i in found_targets:
            i_name = i['name']
            split = i['id'].rsplit('@', 1)
            if len(split) > 1:
                split = split[0].split('@@', 1)
                if len(split) > 1:
                    i_name = split[1]
                else:
                    i_name = split[0]
            p = Path(i['filename'][0]).relative_to(resolved_bdir).parent / i_name
            t = i['type'].replace(' ', '_')
            suggestions.append(f'- ./{p}:{t}')
        suggestions_str = '\n'.join(suggestions)
        raise MesonException(f"Can't invoke target `{target.full_name}`: ambiguous name. Add target type and/or path:\n{suggestions_str}")
    return found_targets[0]