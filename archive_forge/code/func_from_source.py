from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@classmethod
def from_source(cls, filename, args=None, unsaved_files=None, options=0, index=None):
    """Create a TranslationUnit by parsing source.

        This is capable of processing source code both from files on the
        filesystem as well as in-memory contents.

        Command-line arguments that would be passed to clang are specified as
        a list via args. These can be used to specify include paths, warnings,
        etc. e.g. ["-Wall", "-I/path/to/include"].

        In-memory file content can be provided via unsaved_files. This is an
        iterable of 2-tuples. The first element is the filename (str or
        PathLike). The second element defines the content. Content can be
        provided as str source code or as file objects (anything with a read()
        method). If a file object is being used, content will be read until EOF
        and the read cursor will not be reset to its original position.

        options is a bitwise or of TranslationUnit.PARSE_XXX flags which will
        control parsing behavior.

        index is an Index instance to utilize. If not provided, a new Index
        will be created for this TranslationUnit.

        To parse source from the filesystem, the filename of the file to parse
        is specified by the filename argument. Or, filename could be None and
        the args list would contain the filename(s) to parse.

        To parse source from an in-memory buffer, set filename to the virtual
        filename you wish to associate with this source (e.g. "test.c"). The
        contents of that file are then provided in unsaved_files.

        If an error occurs, a TranslationUnitLoadError is raised.

        Please note that a TranslationUnit with parser errors may be returned.
        It is the caller's responsibility to check tu.diagnostics for errors.

        Also note that Clang infers the source language from the extension of
        the input filename. If you pass in source code containing a C++ class
        declaration with the filename "test.c" parsing will fail.
        """
    if args is None:
        args = []
    if unsaved_files is None:
        unsaved_files = []
    if index is None:
        index = Index.create()
    args_array = None
    if len(args) > 0:
        args_array = (c_char_p * len(args))(*[b(x) for x in args])
    unsaved_array = None
    if len(unsaved_files) > 0:
        unsaved_array = (_CXUnsavedFile * len(unsaved_files))()
        for i, (name, contents) in enumerate(unsaved_files):
            if hasattr(contents, 'read'):
                contents = contents.read()
            contents = b(contents)
            unsaved_array[i].name = b(fspath(name))
            unsaved_array[i].contents = contents
            unsaved_array[i].length = len(contents)
    ptr = conf.lib.clang_parseTranslationUnit(index, fspath(filename) if filename is not None else None, args_array, len(args), unsaved_array, len(unsaved_files), options)
    if not ptr:
        raise TranslationUnitLoadError('Error parsing translation unit.')
    return cls(ptr, index=index)