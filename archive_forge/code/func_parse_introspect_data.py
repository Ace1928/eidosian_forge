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
def parse_introspect_data(builddir: Path) -> T.Dict[str, T.List[dict]]:
    """
    Converts a List of name-to-dict to a dict of name-to-dicts (since names are not unique)
    """
    path_to_intro = builddir / 'meson-info' / 'intro-targets.json'
    if not path_to_intro.exists():
        raise MesonException(f'`{path_to_intro.name}` is missing! Directory is not configured yet?')
    with path_to_intro.open(encoding='utf-8') as f:
        schema = json.load(f)
    parsed_data: T.Dict[str, T.List[dict]] = defaultdict(list)
    for target in schema:
        parsed_data[target['name']] += [target]
    return parsed_data