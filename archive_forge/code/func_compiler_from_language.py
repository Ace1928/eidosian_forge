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
def compiler_from_language(env: 'Environment', lang: str, for_machine: MachineChoice) -> T.Optional[Compiler]:
    lang_map: T.Dict[str, T.Callable[['Environment', MachineChoice], Compiler]] = {'c': detect_c_compiler, 'cpp': detect_cpp_compiler, 'objc': detect_objc_compiler, 'cuda': detect_cuda_compiler, 'objcpp': detect_objcpp_compiler, 'java': detect_java_compiler, 'cs': detect_cs_compiler, 'vala': detect_vala_compiler, 'd': detect_d_compiler, 'rust': detect_rust_compiler, 'fortran': detect_fortran_compiler, 'swift': detect_swift_compiler, 'cython': detect_cython_compiler, 'nasm': detect_nasm_compiler, 'masm': detect_masm_compiler}
    return lang_map[lang](env, for_machine) if lang in lang_map else None