import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test_cifti2_metadata():
    md = ci.Cifti2MetaData({'a': 'aval'})
    assert len(md) == 1
    assert list(iter(md)) == ['a']
    assert md['a'] == 'aval'
    assert md.data == dict([('a', 'aval')])
    with pytest.warns(FutureWarning):
        md = ci.Cifti2MetaData(metadata={'a': 'aval'})
    assert md == {'a': 'aval'}
    with pytest.warns(FutureWarning):
        md = ci.Cifti2MetaData(None)
    assert md == {}
    md = ci.Cifti2MetaData()
    assert len(md) == 0
    assert list(iter(md)) == []
    assert md.data == {}
    with pytest.raises(ValueError):
        md.difference_update(None)
    md['a'] = 'aval'
    assert md['a'] == 'aval'
    assert len(md) == 1
    assert md.data == dict([('a', 'aval')])
    del md['a']
    assert len(md) == 0
    metadata_test = [('a', 'aval'), ('b', 'bval')]
    md.update(metadata_test)
    assert md.data == dict(metadata_test)
    assert list(iter(md)) == list(iter(collections.OrderedDict(metadata_test)))
    md.update({'a': 'aval', 'b': 'bval'})
    assert md.data == dict(metadata_test)
    md.update({'a': 'aval', 'd': 'dval'})
    assert md.data == dict(metadata_test + [('d', 'dval')])
    md.difference_update({'a': 'aval', 'd': 'dval'})
    assert md.data == dict(metadata_test[1:])
    with pytest.raises(KeyError):
        md.difference_update({'a': 'aval', 'd': 'dval'})
    assert md.to_xml() == b'<MetaData><MD><Name>b</Name><Value>bval</Value></MD></MetaData>'