import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test_matrixindicesmap():
    mim = ci.Cifti2MatrixIndicesMap(0, 'CIFTI_INDEX_TYPE_LABELS')
    volume = ci.Cifti2Volume()
    volume2 = ci.Cifti2Volume()
    parcel = ci.Cifti2Parcel()
    assert mim.volume is None
    mim.extend((volume, parcel))
    assert mim.volume == volume
    with pytest.raises(ci.Cifti2HeaderError):
        mim.insert(0, volume)
    with pytest.raises(ci.Cifti2HeaderError):
        mim[1] = volume
    mim[0] = volume2
    assert mim.volume == volume2
    del mim.volume
    assert mim.volume is None
    with pytest.raises(ValueError):
        del mim.volume
    mim.volume = volume
    assert mim.volume == volume
    mim.volume = volume2
    assert mim.volume == volume2
    with pytest.raises(ValueError):
        mim.volume = parcel