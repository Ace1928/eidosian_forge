import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def create_scalar_map(applies_to_matrix_dimension):
    maps = [ci.Cifti2NamedMap(name, ci.Cifti2MetaData(meta)) for name, meta in scalars]
    return ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension, 'CIFTI_INDEX_TYPE_SCALARS', maps=maps)