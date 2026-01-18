import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def create_parcel_map(applies_to_matrix_dimension):
    mapping = ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension, 'CIFTI_INDEX_TYPE_PARCELS')
    for name, elements in parcels:
        surfaces = []
        volume = None
        for element in elements:
            if isinstance(element[0], str):
                surfaces.append(ci.Cifti2Vertices(element[0], element[1]))
            else:
                volume = ci.Cifti2VoxelIndicesIJK(element)
        mapping.append(ci.Cifti2Parcel(name, volume, surfaces))
    mapping.extend([ci.Cifti2Surface(f'CIFTI_STRUCTURE_CORTEX_{orientation}', number_of_vertices) for orientation in ['LEFT', 'RIGHT']])
    mapping.volume = ci.Cifti2Volume(dimensions, ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, affine))
    return mapping