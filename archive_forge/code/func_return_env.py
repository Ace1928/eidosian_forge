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
def return_env(self, exists=True):
    """
        Return environment dict.

        Parameters
        ----------
        exists: bool
            It True, only return existing paths.

        Return
        ------
        dict
            environment
        """
    env = dict(include=self._build_paths('include', [self.VCIncludes, self.OSIncludes, self.UCRTIncludes, self.NetFxSDKIncludes], exists), lib=self._build_paths('lib', [self.VCLibraries, self.OSLibraries, self.FxTools, self.UCRTLibraries, self.NetFxSDKLibraries], exists), libpath=self._build_paths('libpath', [self.VCLibraries, self.FxTools, self.VCStoreRefs, self.OSLibpath], exists), path=self._build_paths('path', [self.VCTools, self.VSTools, self.VsTDb, self.SdkTools, self.SdkSetup, self.FxTools, self.MSBuild, self.HTMLHelpWorkshop, self.FSharp], exists))
    if self.vs_ver >= 14 and isfile(self.VCRuntimeRedist):
        env['py_vcruntime_redist'] = self.VCRuntimeRedist
    return env