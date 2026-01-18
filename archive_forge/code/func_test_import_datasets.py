import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_import_datasets(self):
    datasets = robjects.packages.importr('datasets')
    assert isinstance(datasets, robjects.packages.Package)
    assert isinstance(datasets.__rdata__, robjects.packages.PackageData)
    assert isinstance(robjects.packages.data(datasets), robjects.packages.PackageData)