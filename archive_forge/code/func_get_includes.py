from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_includes(self):
    """
        Return an iterable sequence of FileInclusion objects that describe the
        sequence of inclusions in a translation unit. The first object in
        this sequence is always the input file. Note that this method will not
        recursively iterate over header files included through precompiled
        headers.
        """

    def visitor(fobj, lptr, depth, includes):
        if depth > 0:
            loc = lptr.contents
            includes.append(FileInclusion(loc.file, File(fobj), loc, depth))
    includes = []
    conf.lib.clang_getInclusions(self, callbacks['translation_unit_includes'](visitor), includes)
    return iter(includes)