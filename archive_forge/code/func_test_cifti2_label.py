import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test_cifti2_label():
    lb = ci.Cifti2Label()
    lb.label = 'Test'
    lb.key = 0
    assert lb.rgba == (0, 0, 0, 0)
    assert compare_xml_leaf(lb.to_xml().decode('utf-8'), "<Label Key='0' Red='0' Green='0' Blue='0' Alpha='0'>Test</Label>")
    lb.red = 0
    lb.green = 0.1
    lb.blue = 0.2
    lb.alpha = 0.3
    assert lb.rgba == (0, 0.1, 0.2, 0.3)
    assert compare_xml_leaf(lb.to_xml().decode('utf-8'), "<Label Key='0' Red='0' Green='0.1' Blue='0.2' Alpha='0.3'>Test</Label>")
    lb.red = 10
    with pytest.raises(ci.Cifti2HeaderError):
        lb.to_xml()
    lb.red = 0
    lb.key = 'a'
    with pytest.raises(ci.Cifti2HeaderError):
        lb.to_xml()
    lb.key = 0