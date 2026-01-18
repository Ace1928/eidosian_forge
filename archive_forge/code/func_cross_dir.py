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
def cross_dir(self, forcex86=False):
    """
        Cross platform specific subfolder.

        Parameters
        ----------
        forcex86: bool
            Use 'x86' as current architecture even if current architecture is
            not x86.

        Return
        ------
        str
            subfolder: '' if target architecture is current architecture,
            '\\current_target' if not.
        """
    current = 'x86' if forcex86 else self.current_cpu
    return '' if self.target_cpu == current else self.target_dir().replace('\\', '\\%s_' % current)