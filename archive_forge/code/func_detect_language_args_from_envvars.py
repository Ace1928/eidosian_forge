from __future__ import annotations
import sys, os, subprocess, shutil
import shlex
import typing as T
from .. import envconfig
from .. import mlog
from ..compilers import compilers
from ..compilers.detect import defaults as compiler_names
def detect_language_args_from_envvars(langname: str, envvar_suffix: str='') -> T.Tuple[T.List[str], T.List[str]]:
    ldflags = tuple(shlex.split(os.environ.get('LDFLAGS' + envvar_suffix, '')))
    compile_args = []
    if langname in compilers.CFLAGS_MAPPING:
        compile_args = shlex.split(os.environ.get(compilers.CFLAGS_MAPPING[langname] + envvar_suffix, ''))
    if langname in compilers.LANGUAGES_USING_CPPFLAGS:
        cppflags = tuple(shlex.split(os.environ.get('CPPFLAGS' + envvar_suffix, '')))
        lang_compile_args = list(cppflags) + compile_args
    else:
        lang_compile_args = compile_args
    lang_link_args = list(ldflags) + compile_args
    return (lang_compile_args, lang_link_args)