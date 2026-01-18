import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_datatsets_names(self):
    datasets = robjects.packages.importr('datasets')
    datasets_data = robjects.packages.data(datasets)
    datasets_names = tuple(datasets_data.names())
    assert len(datasets_names) > 0
    assert all((isinstance(x, str) for x in datasets_names))