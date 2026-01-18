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
def target_is_x86(self):
    """
        Return True if target CPU is x86 32 bits..

        Return
        ------
        bool
            CPU is x86 32 bits
        """
    return self.target_cpu == 'x86'