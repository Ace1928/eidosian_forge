import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_installedpackages():
    instapacks = robjects.packages.InstalledPackages()
    res = instapacks.isinstalled('foo')
    assert res is False
    ncols = len(instapacks.colnames)
    for row in instapacks:
        assert ncols == len(row)