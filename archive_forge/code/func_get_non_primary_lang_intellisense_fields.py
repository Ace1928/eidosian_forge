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
def get_non_primary_lang_intellisense_fields(vslite_ctx: dict, target_id: str, primary_src_lang: str) -> T.Dict[str, T.Dict[str, T.Tuple[str, str, str]]]:
    defs_paths_opts_per_lang_and_buildtype = {}
    for buildtype in coredata.get_genvs_default_buildtype_list():
        captured_build_args = vslite_ctx[buildtype][target_id]
        non_primary_build_args_per_src_lang = [(lang, build_args) for lang, build_args in captured_build_args.items() if lang != primary_src_lang]
        for src_lang, args_list in non_primary_build_args_per_src_lang:
            if src_lang not in defs_paths_opts_per_lang_and_buildtype:
                defs_paths_opts_per_lang_and_buildtype[src_lang] = {}
            defs_paths_opts_per_lang_and_buildtype[src_lang][buildtype] = Vs2010Backend._extract_nmake_fields(args_list)
    return defs_paths_opts_per_lang_and_buildtype