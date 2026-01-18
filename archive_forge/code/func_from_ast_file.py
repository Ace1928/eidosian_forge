from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@classmethod
def from_ast_file(cls, filename, index=None):
    """Create a TranslationUnit instance from a saved AST file.

        A previously-saved AST file (provided with -emit-ast or
        TranslationUnit.save()) is loaded from the filename specified.

        If the file cannot be loaded, a TranslationUnitLoadError will be
        raised.

        index is optional and is the Index instance to use. If not provided,
        a default Index will be created.

        filename can be str or PathLike.
        """
    if index is None:
        index = Index.create()
    ptr = conf.lib.clang_createTranslationUnit(index, fspath(filename))
    if not ptr:
        raise TranslationUnitLoadError(filename)
    return cls(ptr=ptr, index=index)