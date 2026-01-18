from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def reparse(self, unsaved_files=None, options=0):
    """
        Reparse an already parsed translation unit.

        In-memory contents for files can be provided by passing a list of pairs
        as unsaved_files, the first items should be the filenames to be mapped
        and the second should be the contents to be substituted for the
        file. The contents may be passed as strings or file objects.
        """
    if unsaved_files is None:
        unsaved_files = []
    unsaved_files_array = 0
    if len(unsaved_files):
        unsaved_files_array = (_CXUnsavedFile * len(unsaved_files))()
        for i, (name, contents) in enumerate(unsaved_files):
            if hasattr(contents, 'read'):
                contents = contents.read()
            contents = b(contents)
            unsaved_files_array[i].name = b(fspath(name))
            unsaved_files_array[i].contents = contents
            unsaved_files_array[i].length = len(contents)
    ptr = conf.lib.clang_reparseTranslationUnit(self, len(unsaved_files), unsaved_files_array, options)