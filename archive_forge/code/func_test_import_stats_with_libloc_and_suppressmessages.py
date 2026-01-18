import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_import_stats_with_libloc_and_suppressmessages(self):
    path = os.path.dirname(robjects.packages_utils.get_packagepath('stats'))
    stats = robjects.packages.importr('stats', lib_loc=path, on_conflict='warn', suppress_messages=False)
    assert isinstance(stats, robjects.packages.Package)