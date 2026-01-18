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
def target_dir(self, hidex86=False, x64=False):
    """
        Target platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\\x86' if architecture is x86.
        x64: bool
            return '\\x64' and not '\\amd64' if architecture is amd64.

        Return
        ------
        str
            subfolder: '\\current', or '' (see hidex86 parameter)
        """
    return '' if self.target_cpu == 'x86' and hidex86 else '\\x64' if self.target_cpu == 'amd64' and x64 else '\\%s' % self.target_cpu