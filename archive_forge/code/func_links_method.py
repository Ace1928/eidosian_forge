from __future__ import annotations
import collections
import enum
import functools
import os
import itertools
import typing as T
from .. import build
from .. import coredata
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..compilers import SUFFIX_TO_LANG
from ..compilers.compilers import CompileCheckMode
from ..interpreterbase import (ObjectHolder, noPosargs, noKwargs,
from ..interpreterbase.decorators import ContainerTypeInfo, typed_kwargs, KwargInfo, typed_pos_args
from ..mesonlib import OptionKey
from .interpreterobjects import (extract_required_kwarg, extract_search_dirs)
from .type_checking import REQUIRED_KW, in_set_validator, NoneType
@typed_pos_args('compiler.links', (str, mesonlib.File))
@typed_kwargs('compiler.links', *_COMPILES_KWS)
def links_method(self, args: T.Tuple['mesonlib.FileOrString'], kwargs: 'CompileKW') -> bool:
    code = args[0]
    compiler = None
    if isinstance(code, mesonlib.File):
        if code.is_built:
            FeatureNew.single_use('compiler.links with file created at setup time', '1.2.0', self.subproject, 'It was broken and either errored or returned false.', self.current_node)
        self.interpreter.add_build_def_file(code)
        code = mesonlib.File.from_absolute_file(code.absolute_path(self.environment.source_dir, self.environment.build_dir))
        suffix = code.suffix
        if suffix not in self.compiler.file_suffixes:
            for_machine = self.compiler.for_machine
            clist = self.interpreter.coredata.compilers[for_machine]
            if suffix not in SUFFIX_TO_LANG:
                mlog.warning(f'Unknown suffix for test file {code}')
            elif SUFFIX_TO_LANG[suffix] not in clist:
                mlog.warning(f'Passed {SUFFIX_TO_LANG[suffix]} source to links method, not specified for {for_machine.get_lower_case_name()} machine.')
            else:
                compiler = clist[SUFFIX_TO_LANG[suffix]]
    testname = kwargs['name']
    extra_args = functools.partial(self._determine_args, kwargs)
    deps, msg = self._determine_dependencies(kwargs['dependencies'], compile_only=False)
    result, cached = self.compiler.links(code, self.environment, compiler=compiler, extra_args=extra_args, dependencies=deps)
    cached_msg = mlog.blue('(cached)') if cached else ''
    if testname:
        if result:
            h = mlog.green('YES')
        else:
            h = mlog.red('NO')
        mlog.log('Checking if', mlog.bold(testname, True), msg, 'links:', h, cached_msg)
    return result