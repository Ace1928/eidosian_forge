from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
def create_extension_list(patterns, exclude=None, ctx=None, aliases=None, quiet=False, language=None, exclude_failures=False):
    if language is not None:
        print('Warning: passing language={0!r} to cythonize() is deprecated. Instead, put "# distutils: language={0}" in your .pyx or .pxd file(s)'.format(language))
    if exclude is None:
        exclude = []
    if patterns is None:
        return ([], {})
    elif isinstance(patterns, basestring) or not isinstance(patterns, Iterable):
        patterns = [patterns]
    from distutils.extension import Extension
    if 'setuptools' in sys.modules:
        extension_classes = (Extension, sys.modules['setuptools.extension']._Extension, sys.modules['setuptools'].Extension)
    else:
        extension_classes = (Extension,)
    explicit_modules = {m.name for m in patterns if isinstance(m, extension_classes)}
    deps = create_dependency_tree(ctx, quiet=quiet)
    to_exclude = set()
    if not isinstance(exclude, list):
        exclude = [exclude]
    for pattern in exclude:
        to_exclude.update(map(os.path.abspath, extended_iglob(pattern)))
    module_list = []
    module_metadata = {}
    create_extension = ctx.options.create_extension or default_create_extension
    seen = set()
    for pattern in patterns:
        if not isinstance(pattern, extension_classes):
            pattern = encode_filename_in_py2(pattern)
        if isinstance(pattern, str):
            filepattern = pattern
            template = Extension(pattern, [])
            name = '*'
            base = None
            ext_language = language
        elif isinstance(pattern, extension_classes):
            cython_sources = [s for s in pattern.sources if os.path.splitext(s)[1] in ('.py', '.pyx')]
            if cython_sources:
                filepattern = cython_sources[0]
                if len(cython_sources) > 1:
                    print(u"Warning: Multiple cython sources found for extension '%s': %s\nSee https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html for sharing declarations among Cython files." % (pattern.name, cython_sources))
            else:
                module_list.append(pattern)
                continue
            template = pattern
            name = template.name
            base = DistutilsInfo(exn=template)
            ext_language = None
        else:
            msg = str('pattern is not of type str nor subclass of Extension (%s) but of type %s and class %s' % (repr(Extension), type(pattern), pattern.__class__))
            raise TypeError(msg)
        for file in nonempty(sorted(extended_iglob(filepattern)), "'%s' doesn't match any files" % filepattern):
            if os.path.abspath(file) in to_exclude:
                continue
            module_name = deps.fully_qualified_name(file)
            if '*' in name:
                if module_name in explicit_modules:
                    continue
            elif name:
                module_name = name
            Utils.raise_error_if_module_name_forbidden(module_name)
            if module_name not in seen:
                try:
                    kwds = deps.distutils_info(file, aliases, base).values
                except Exception:
                    if exclude_failures:
                        continue
                    raise
                if base is not None:
                    for key, value in base.values.items():
                        if key not in kwds:
                            kwds[key] = value
                kwds['name'] = module_name
                sources = [file] + [m for m in template.sources if m != filepattern]
                if 'sources' in kwds:
                    for source in kwds['sources']:
                        source = encode_filename_in_py2(source)
                        if source not in sources:
                            sources.append(source)
                kwds['sources'] = sources
                if ext_language and 'language' not in kwds:
                    kwds['language'] = ext_language
                np_pythran = kwds.pop('np_pythran', False)
                m, metadata = create_extension(template, kwds)
                m.np_pythran = np_pythran or getattr(m, 'np_pythran', False)
                if m.np_pythran:
                    update_pythran_extension(m)
                module_list.append(m)
                module_metadata[module_name] = metadata
                if file not in m.sources:
                    target_file = os.path.splitext(file)[0] + ('.cpp' if m.language == 'c++' else '.c')
                    try:
                        m.sources.remove(target_file)
                    except ValueError:
                        print(u'Warning: Cython source file not found in sources list, adding %s' % file)
                    m.sources.insert(0, file)
                seen.add(name)
    return (module_list, module_metadata)