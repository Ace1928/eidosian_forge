import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_new_with_dot_conflict2(self):
    env = robjects.Environment()
    name_in_use = dir(packages.Package(env, 'foo'))[0]
    env[name_in_use] = robjects.StrVector('abcd')
    with pytest.raises(packages.LibraryError):
        robjects.packages.Package(env, 'dummy_package')