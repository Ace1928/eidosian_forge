import io
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from packaging.version import Version
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader
from nibabel.tests import test_nifti2 as tn2
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tmpdirs import InTemporaryDirectory
@needs_nibabel_data('nitest-cifti2')
def test_cifti2types():
    """Check that we instantiate Cifti2 classes correctly, and that our
    test files exercise all classes"""
    counter = {ci.Cifti2LabelTable: 0, ci.Cifti2Label: 0, ci.Cifti2NamedMap: 0, ci.Cifti2Surface: 0, ci.Cifti2VoxelIndicesIJK: 0, ci.Cifti2Vertices: 0, ci.Cifti2Parcel: 0, ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ: 0, ci.Cifti2Volume: 0, ci.Cifti2VertexIndices: 0, ci.Cifti2BrainModel: 0, ci.Cifti2MatrixIndicesMap: 0}
    for name in datafiles:
        hdr = ci.load(name).header
        assert isinstance(hdr.matrix, ci.Cifti2Matrix)
        assert isinstance(hdr.matrix.metadata, ci.Cifti2MetaData)
        for mim in hdr.matrix:
            assert isinstance(mim, ci.Cifti2MatrixIndicesMap)
            counter[ci.Cifti2MatrixIndicesMap] += 1
            for map_ in mim:
                print(map_)
                if isinstance(map_, ci.Cifti2BrainModel):
                    counter[ci.Cifti2BrainModel] += 1
                    if isinstance(map_.vertex_indices, ci.Cifti2VertexIndices):
                        counter[ci.Cifti2VertexIndices] += 1
                    if isinstance(map_.voxel_indices_ijk, ci.Cifti2VoxelIndicesIJK):
                        counter[ci.Cifti2VoxelIndicesIJK] += 1
                elif isinstance(map_, ci.Cifti2NamedMap):
                    counter[ci.Cifti2NamedMap] += 1
                    assert isinstance(map_.metadata, ci.Cifti2MetaData)
                    if isinstance(map_.label_table, ci.Cifti2LabelTable):
                        counter[ci.Cifti2LabelTable] += 1
                        for label in map_.label_table:
                            assert isinstance(map_.label_table[label], ci.Cifti2Label)
                            counter[ci.Cifti2Label] += 1
                elif isinstance(map_, ci.Cifti2Parcel):
                    counter[ci.Cifti2Parcel] += 1
                    if isinstance(map_.voxel_indices_ijk, ci.Cifti2VoxelIndicesIJK):
                        counter[ci.Cifti2VoxelIndicesIJK] += 1
                    assert isinstance(map_.vertices, list)
                    for vtcs in map_.vertices:
                        assert isinstance(vtcs, ci.Cifti2Vertices)
                        counter[ci.Cifti2Vertices] += 1
                elif isinstance(map_, ci.Cifti2Surface):
                    counter[ci.Cifti2Surface] += 1
                elif isinstance(map_, ci.Cifti2Volume):
                    counter[ci.Cifti2Volume] += 1
                    if isinstance(map_.transformation_matrix_voxel_indices_ijk_to_xyz, ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ):
                        counter[ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ] += 1
            assert list(mim.named_maps) == [m_ for m_ in mim if isinstance(m_, ci.Cifti2NamedMap)]
            assert list(mim.surfaces) == [m_ for m_ in mim if isinstance(m_, ci.Cifti2Surface)]
            assert list(mim.parcels) == [m_ for m_ in mim if isinstance(m_, ci.Cifti2Parcel)]
            assert list(mim.brain_models) == [m_ for m_ in mim if isinstance(m_, ci.Cifti2BrainModel)]
            assert ([mim.volume] if mim.volume else []) == [m_ for m_ in mim if isinstance(m_, ci.Cifti2Volume)]
    for klass, count in counter.items():
        assert count > 0, 'No exercise of ' + klass.__name__