import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test_cifti2_parcel():
    pl = ci.Cifti2Parcel()
    with pytest.raises(ci.Cifti2HeaderError):
        pl.to_xml()
    with pytest.raises(TypeError):
        pl.append_cifti_vertices(None)
    with pytest.raises(ValueError):
        ci.Cifti2Parcel(vertices=[1, 2, 3])
    pl = ci.Cifti2Parcel(name='region', voxel_indices_ijk=ci.Cifti2VoxelIndicesIJK([[1, 2, 3]]), vertices=[ci.Cifti2Vertices([0, 1, 2])])
    pl.pop_cifti2_vertices(0)
    assert len(pl.vertices) == 0
    assert pl.to_xml() == b'<Parcel Name="region"><VoxelIndicesIJK>1 2 3</VoxelIndicesIJK></Parcel>'