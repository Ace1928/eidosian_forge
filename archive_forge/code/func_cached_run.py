from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
def cached_run(self, code: str, env: 'Environment', *, extra_args: T.Union[T.List[str], T.Callable[[CompileCheckMode], T.List[str]], None]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> RunResult:
    run_check_cache = env.coredata.run_check_cache
    args = self.build_wrapper_args(env, extra_args, dependencies, CompileCheckMode('link'))
    key = (code, tuple(args))
    if key in run_check_cache:
        p = run_check_cache[key]
        p.cached = True
        mlog.debug('Using cached run result:')
        mlog.debug('Code:\n', code)
        mlog.debug('Args:\n', extra_args)
        mlog.debug('Cached run returncode:\n', p.returncode)
        mlog.debug('Cached run stdout:\n', p.stdout)
        mlog.debug('Cached run stderr:\n', p.stderr)
    else:
        p = self.run(code, env, extra_args=extra_args, dependencies=dependencies)
        run_check_cache[key] = p
    return p