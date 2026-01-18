import os
import pytest
import nipype.testing as npt
from nipype.testing import example_data
import numpy as np
from nipype.algorithms import mesh as m
from ...interfaces import vtkbase as VTKInfo
@pytest.mark.skipif(VTKInfo.no_tvtk(), reason='tvtk is not installed')
def test_meshwarpmaths(tmpdir):
    tmpdir.chdir()