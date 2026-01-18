from __future__ import annotations
from ..mesonlib import (
from ..envconfig import BinaryTable
from .. import mlog
from ..linkers import guess_win_linker, guess_nix_linker
import subprocess
import platform
import re
import shutil
import tempfile
import os
import typing as T
def detect_compiler_for(env: 'Environment', lang: str, for_machine: MachineChoice, skip_sanity_check: bool, subproject: str) -> T.Optional[Compiler]:
    comp = compiler_from_language(env, lang, for_machine)
    if comp is None:
        return comp
    assert comp.for_machine == for_machine
    env.coredata.process_compiler_options(lang, comp, env, subproject)
    if not skip_sanity_check:
        comp.sanity_check(env.get_scratch_dir(), env)
    env.coredata.compilers[comp.for_machine][lang] = comp
    return comp