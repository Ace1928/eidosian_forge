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
@property
def netfx_sdk(self):
    """
        Microsoft .NET Framework SDK registry key.

        Return
        ------
        str
            Registry key
        """
    return join(self.microsoft_sdk, 'NETFXSDK')