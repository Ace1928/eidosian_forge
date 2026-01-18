import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test__float_01():
    assert _float_01(0) == 0
    assert _float_01(1) == 1
    assert _float_01('0') == 0
    assert _float_01('0.2') == 0.2
    with pytest.raises(ValueError):
        _float_01(1.1)
    with pytest.raises(ValueError):
        _float_01(-0.1)
    with pytest.raises(ValueError):
        _float_01(2)
    with pytest.raises(ValueError):
        _float_01(-1)
    with pytest.raises(ValueError):
        _float_01('foo')