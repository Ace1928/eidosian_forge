import glob
import inspect
import logging
import os
import platform
import importlib.util
import sys
from . import envvar
from .dependencies import ctypes
from .deprecation import deprecated, relocated_module_attribute
def find_path(name, validate, cwd=True, mode=os.R_OK, ext=None, pathlist=[], allow_pathlist_deep_references=True):
    """Locate a path, given a set of search parameters

    Parameters
    ----------
    name : str
        The name to locate.  The name may contain references to a user's
        home directory (``~user``), environment variables
        (``${HOME}/bin``), and shell wildcards (``?`` and ``*``); all of
        which will be expanded.

    validate : function
        A function to call to validate the path (used by find_file and
        find_dir to discriminate files and directories)

    cwd : bool
        Start by looking in the current working directory
        [default: True]

    mode : mask
        If not None, only return entries that can be accessed for
        reading/writing/executing.  Valid values are the inclusive OR of
        {os.R_OK, os.W_OK, os.X_OK} [default: ``os.R_OK``]

    ext : str or iterable of str
        If not None, also look for name+ext [default: None]

    pathlist : str or iterable of str
        A list of strings containing paths to search, each string
        contains a single path.  If pathlist is a string, then it is
        first split using os.pathsep to generate the pathlist
        [default: ``[]``].

    allow_pathlist_deep_references : bool
       If allow_pathlist_deep_references is True and the name
       appears to be a relative path, allow deep reference matches
       relative to directories in the pathlist (e.g., if name is
       ``foo/my.exe`` and ``/usr/bin`` is in the pathlist, then
       :py:func:`find_file` could return ``/usr/bin/foo/my.exe``).  If
       allow_pathlist_deep_references is False and the name appears
       to be a relative path, then only matches relative to the current
       directory are allowed (assuming cwd==True).  [default: True]

    Notes
    -----
        find_path uses glob, so the path and/or name may contain
        wildcards.  The first matching entry is returned.

    """
    name = os.path.expanduser(os.path.expandvars(name))
    locations = []
    if cwd:
        locations.append(os.getcwd())
    if allow_pathlist_deep_references or os.path.basename(name) == name:
        if isinstance(pathlist, str):
            locations.extend(pathlist.split(os.pathsep))
        else:
            locations.extend(pathlist)
    extlist = ['']
    if ext:
        if isinstance(ext, str):
            extlist.append(ext)
        else:
            extlist.extend(ext)
    for path in locations:
        if not path:
            continue
        for _ext in extlist:
            for test in glob.glob(os.path.join(path, name + _ext)):
                if not validate(test):
                    continue
                if mode is not None and (not os.access(test, mode)):
                    continue
                return os.path.abspath(test)
    return None