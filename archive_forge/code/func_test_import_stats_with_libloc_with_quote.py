import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_import_stats_with_libloc_with_quote(self):
    path = 'coin"coin'
    with pytest.raises(robjects.packages.PackageNotInstalledError), pytest.warns(UserWarning):
        Tmp_File = io.StringIO
        tmp_file = Tmp_File()
        try:
            stdout = sys.stdout
            sys.stdout = tmp_file
            robjects.packages.importr('dummy_inexistant', lib_loc=path)
        finally:
            sys.stdout = stdout
            tmp_file.close()