import os
import pytest
import nipype.testing as npt
from nipype.testing import example_data
import numpy as np
from nipype.algorithms import mesh as m
from ...interfaces import vtkbase as VTKInfo
@pytest.mark.skipif(VTKInfo.no_tvtk(), reason='tvtk is not installed')
def test_ident_distances(tmpdir):
    tmpdir.chdir()
    in_surf = example_data('surf01.vtk')
    dist_ident = m.ComputeMeshWarp()
    dist_ident.inputs.surface1 = in_surf
    dist_ident.inputs.surface2 = in_surf
    dist_ident.inputs.out_file = tmpdir.join('distance.npy').strpath
    res = dist_ident.run()
    assert res.outputs.distance == 0.0
    dist_ident.inputs.weighting = 'area'
    res = dist_ident.run()
    assert res.outputs.distance == 0.0