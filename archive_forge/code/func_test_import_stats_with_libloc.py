import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_import_stats_with_libloc(self):
    path = os.path.dirname(robjects.packages_utils.get_packagepath('stats'))
    stats = robjects.packages.importr('stats', on_conflict='warn', lib_loc=path)
    assert isinstance(stats, robjects.packages.Package)