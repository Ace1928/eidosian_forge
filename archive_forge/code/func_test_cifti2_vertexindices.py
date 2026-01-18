import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test_cifti2_vertexindices():
    vi = ci.Cifti2VertexIndices()
    assert len(vi) == 0
    with pytest.raises(ci.Cifti2HeaderError):
        vi.to_xml()
    vi.extend(np.array([0, 1, 2]))
    assert len(vi) == 3
    assert vi.to_xml() == b'<VertexIndices>0 1 2</VertexIndices>'
    with pytest.raises(ValueError):
        vi[0] = 'a'
    vi[0] = 10
    assert vi[0] == 10
    assert len(vi) == 3