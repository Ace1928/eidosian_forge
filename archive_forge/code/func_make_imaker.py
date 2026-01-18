import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def make_imaker(self, arr, header=None, ni_header=None):
    for idx, sz in enumerate(arr.shape):
        maps = [ci.Cifti2NamedMap(str(value)) for value in range(sz)]
        mim = ci.Cifti2MatrixIndicesMap((idx,), 'CIFTI_INDEX_TYPE_SCALARS', maps=maps)
        header.matrix.append(mim)
    return lambda: self.image_maker(arr.copy(), header, ni_header)