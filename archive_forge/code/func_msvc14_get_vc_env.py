import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
def msvc14_get_vc_env(plat_spec):
    """
    Patched "distutils._msvccompiler._get_vc_env" for support extra
    Microsoft Visual C++ 14.X compilers.

    Set environment without use of "vcvarsall.bat".

    Parameters
    ----------
    plat_spec: str
        Target architecture.

    Return
    ------
    dict
        environment
    """
    try:
        return _msvc14_get_vc_env(plat_spec)
    except distutils.errors.DistutilsPlatformError as exc:
        _augment_exception(exc, 14.0)
        raise