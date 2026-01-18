from __future__ import annotations
import os
import shutil
import typing as T
from ... import mesonlib
from ...linkers.linkers import AppleDynamicLinker, ClangClDynamicLinker, LLVMDynamicLinker, GnuGoldDynamicLinker, \
from ...mesonlib import OptionKey
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
def get_lto_link_args(self, *, threads: int=0, mode: str='default', thinlto_cache_dir: T.Optional[str]=None) -> T.List[str]:
    args = self.get_lto_compile_args(threads=threads, mode=mode)
    if mode == 'thin' and thinlto_cache_dir is not None:
        args.extend(self.linker.get_thinlto_cache_args(thinlto_cache_dir))
    if threads > 0:
        if not mesonlib.version_compare(self.version, '>=4.0.0'):
            raise mesonlib.MesonException('clang support for LTO threads requires clang >=4.0')
        args.append(f'-flto-jobs={threads}')
    return args