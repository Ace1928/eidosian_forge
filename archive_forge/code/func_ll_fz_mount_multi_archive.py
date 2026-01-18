from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_mount_multi_archive(arch_, sub, path):
    """
    Low-level wrapper for `::fz_mount_multi_archive()`.
    Add an archive to the set of archives handled by a multi
    archive.

    If path is NULL, then the archive contents will appear at the
    top level, otherwise, the archives contents will appear prefixed
    by path.
    """
    return _mupdf.ll_fz_mount_multi_archive(arch_, sub, path)