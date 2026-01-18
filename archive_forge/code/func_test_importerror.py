import os
import pytest
import nipype.testing as npt
from nipype.testing import example_data
import numpy as np
from nipype.algorithms import mesh as m
from ...interfaces import vtkbase as VTKInfo
@pytest.mark.skipif(not VTKInfo.no_tvtk(), reason='tvtk is installed')
def test_importerror():
    with pytest.raises(ImportError):
        m.ComputeMeshWarp()
    with pytest.raises(ImportError):
        m.WarpPoints()
    with pytest.raises(ImportError):
        m.MeshWarpMaths()