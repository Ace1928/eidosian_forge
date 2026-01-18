import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test_cifti2_vertices():
    vs = ci.Cifti2Vertices()
    with pytest.raises(ci.Cifti2HeaderError):
        vs.to_xml()
    vs.brain_structure = 'CIFTI_STRUCTURE_OTHER'
    assert vs.to_xml() == b'<Vertices BrainStructure="CIFTI_STRUCTURE_OTHER" />'
    assert len(vs) == 0
    vs.extend(np.array([0, 1, 2]))
    assert len(vs) == 3
    with pytest.raises(ValueError):
        vs[1] = 'a'
    with pytest.raises(ValueError):
        vs.insert(1, 'a')
    assert vs.to_xml() == b'<Vertices BrainStructure="CIFTI_STRUCTURE_OTHER">0 1 2</Vertices>'
    vs[0] = 10
    assert vs[0] == 10
    assert len(vs) == 3
    vs = ci.Cifti2Vertices(vertices=[0, 1, 2])
    assert len(vs) == 3