import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_metadata_list_interface():
    md = GiftiMetaData(key='value')
    with pytest.warns(DeprecationWarning):
        mdlist = md.data
    assert len(mdlist) == 1
    assert mdlist[0].name == 'key'
    assert mdlist[0].value == 'value'
    mdlist[0].name = 'foo'
    assert mdlist[0].name == 'foo'
    assert 'foo' in md
    assert 'key' not in md
    assert md['foo'] == 'value'
    mdlist[0].value = 'bar'
    assert mdlist[0].value == 'bar'
    assert md['foo'] == 'bar'
    with pytest.warns(DeprecationWarning) as w:
        nvpair = GiftiNVPairs('key', 'value')
    mdlist.append(nvpair)
    assert len(mdlist) == 2
    assert mdlist[1].name == 'key'
    assert mdlist[1].value == 'value'
    assert len(md) == 2
    assert md == {'foo': 'bar', 'key': 'value'}
    mdlist.clear()
    assert len(mdlist) == 0
    assert len(md) == 0
    with pytest.warns(DeprecationWarning) as w:
        foobar = GiftiNVPairs('foo', 'bar')
    mdlist.extend([nvpair, foobar])
    assert len(mdlist) == 2
    assert len(md) == 2
    assert md == {'key': 'value', 'foo': 'bar'}
    with pytest.warns(DeprecationWarning) as w:
        lastone = GiftiNVPairs('last', 'one')
    mdlist.insert(1, lastone)
    assert len(mdlist) == 3
    assert len(md) == 3
    assert mdlist[1].name == 'last'
    assert mdlist[1].value == 'one'
    assert md == {'key': 'value', 'foo': 'bar', 'last': 'one'}
    mypair = mdlist.pop(0)
    assert isinstance(mypair, GiftiNVPairs)
    assert mypair.name == 'key'
    assert mypair.value == 'value'
    assert len(mdlist) == 2
    assert len(md) == 2
    assert 'key' not in md
    assert md == {'foo': 'bar', 'last': 'one'}
    mypair.name = 'completelynew'
    mypair.value = 'strings'
    assert 'completelynew' not in md
    assert md == {'foo': 'bar', 'last': 'one'}
    lastpair = mdlist.pop()
    assert len(mdlist) == 1
    assert len(md) == 1
    assert md == {'last': 'one'}
    with pytest.warns(DeprecationWarning) as w:
        lastoneagain = GiftiNVPairs('last', 'one')
    mdlist.remove(lastoneagain)
    assert len(mdlist) == 0
    assert len(md) == 0