from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
def process_compilers(self) -> T.List[str]:
    """
        Populate self.compilers, which is the list of compilers that this
        target will use for compiling all its sources.
        We also add compilers that were used by extracted objects to simplify
        dynamic linker determination.
        Returns a list of missing languages that we can add implicitly, such as
        C/C++ compiler for cython.
        """
    missing_languages: T.List[str] = []
    if not any([self.sources, self.generated, self.objects, self.structured_sources]):
        return missing_languages
    sources: T.List['FileOrString'] = list(self.sources)
    generated = self.generated.copy()
    if self.structured_sources:
        for v in self.structured_sources.sources.values():
            for src in v:
                if isinstance(src, (str, File)):
                    sources.append(src)
                else:
                    generated.append(src)
    for gensrc in generated:
        for s in gensrc.get_outputs():
            if not is_object(s):
                sources.append(s)
    for d in self.external_deps:
        for s in d.sources:
            if isinstance(s, (str, File)):
                sources.append(s)
    for o in self.objects:
        if not isinstance(o, ExtractedObjects):
            continue
        compsrcs = o.classify_all_sources(o.srclist, [])
        for comp in compsrcs:
            if comp.language == 'vala':
                continue
            if comp.language not in self.compilers:
                self.compilers[comp.language] = comp
    if sources:
        for s in sources:
            for lang, compiler in self.all_compilers.items():
                if compiler.can_compile(s):
                    if lang not in self.compilers:
                        self.compilers[lang] = compiler
                    break
            else:
                if is_known_suffix(s):
                    path = pathlib.Path(str(s)).as_posix()
                    m = f'No {self.for_machine.get_lower_case_name()} machine compiler for {path!r}'
                    raise MesonException(m)
    if 'vala' in self.compilers and 'c' not in self.compilers:
        self.compilers['c'] = self.all_compilers['c']
    if 'cython' in self.compilers:
        key = OptionKey('language', machine=self.for_machine, lang='cython')
        value = self.get_option(key)
        try:
            self.compilers[value] = self.all_compilers[value]
        except KeyError:
            missing_languages.append(value)
    return missing_languages