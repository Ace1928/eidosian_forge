import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def create_geometry_map(applies_to_matrix_dimension):
    voxels = ci.Cifti2VoxelIndicesIJK(brain_models[0][1])
    left_thalamus = ci.Cifti2BrainModel(index_offset=0, index_count=4, model_type='CIFTI_MODEL_TYPE_VOXELS', brain_structure=brain_models[0][0], voxel_indices_ijk=voxels)
    vertices = ci.Cifti2VertexIndices(np.array(brain_models[1][1]))
    left_cortex = ci.Cifti2BrainModel(index_offset=4, index_count=5, model_type='CIFTI_MODEL_TYPE_SURFACE', brain_structure=brain_models[1][0], vertex_indices=vertices)
    left_cortex.surface_number_of_vertices = number_of_vertices
    vertices = ci.Cifti2VertexIndices(np.array(brain_models[2][1]))
    right_cortex = ci.Cifti2BrainModel(index_offset=9, index_count=1, model_type='CIFTI_MODEL_TYPE_SURFACE', brain_structure=brain_models[2][0], vertex_indices=vertices)
    right_cortex.surface_number_of_vertices = number_of_vertices
    volume = ci.Cifti2Volume(dimensions, ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, affine))
    return ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension, 'CIFTI_INDEX_TYPE_BRAIN_MODELS', maps=[left_thalamus, left_cortex, right_cortex, volume])