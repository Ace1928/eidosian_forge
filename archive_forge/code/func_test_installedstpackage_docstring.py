import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_installedstpackage_docstring():
    stats = robjects.packages.importr('stats', on_conflict='warn')
    assert stats.__doc__.startswith('Python representation of an R package.')