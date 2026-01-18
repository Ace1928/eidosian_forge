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
def microsoft(self, key, x86=False):
    """
        Return key in Microsoft software registry.

        Parameters
        ----------
        key: str
            Registry key path where look.
        x86: str
            Force x86 software registry.

        Return
        ------
        str
            Registry key
        """
    node64 = '' if self.pi.current_is_x86() or x86 else 'Wow6432Node'
    return join('Software', node64, 'Microsoft', key)