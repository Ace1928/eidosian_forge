import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_new_with_dot_conflict(self):
    env = robjects.Environment()
    env['a.a_a'] = robjects.StrVector('abcd')
    env['a_a.a'] = robjects.IntVector((1, 2, 3))
    env['c'] = robjects.r(' function(x) x^2')
    with pytest.raises(packages.LibraryError):
        robjects.packages.Package(env, 'dummy_package')