import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test_cifti2_labeltable():
    lt = ci.Cifti2LabelTable()
    assert len(lt) == 0
    with pytest.raises(ci.Cifti2HeaderError):
        lt.to_xml()
    with pytest.raises(ci.Cifti2HeaderError):
        lt._to_xml_element()
    label = ci.Cifti2Label(label='Test', key=0)
    lt[0] = label
    assert len(lt) == 1
    assert dict(lt) == {label.key: label}
    lt.clear()
    lt.append(label)
    assert len(lt) == 1
    assert dict(lt) == {label.key: label}
    lt.clear()
    test_tuple = (label.label, label.red, label.green, label.blue, label.alpha)
    lt[label.key] = test_tuple
    assert len(lt) == 1
    v = lt[label.key]
    assert (v.label, v.red, v.green, v.blue, v.alpha) == test_tuple
    with pytest.raises(ValueError):
        lt[1] = label
    with pytest.raises(ValueError):
        lt[0] = test_tuple[:-1]
    with pytest.raises(ValueError):
        lt[0] = ('foo', 1.1, 0, 0, 1)
    with pytest.raises(ValueError):
        lt[0] = ('foo', 1.0, -1, 0, 1)
    with pytest.raises(ValueError):
        lt[0] = ('foo', 1.0, 0, -0.1, 1)