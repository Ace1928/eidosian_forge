import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
def test_ext_scalar_from_array():
    data = [b'0123456789abcdef', b'0123456789abcdef', b'zyxwvutsrqponmlk', None]
    storage = pa.array(data, type=pa.binary(16))
    ty1 = UuidType()
    ty2 = ParamExtType(16)
    ty3 = UuidType2()
    a = pa.ExtensionArray.from_storage(ty1, storage)
    b = pa.ExtensionArray.from_storage(ty2, storage)
    c = pa.ExtensionArray.from_storage(ty3, storage)
    scalars_a = list(a)
    assert len(scalars_a) == 4
    assert ty1.__arrow_ext_scalar_class__() == UuidScalarType
    assert isinstance(a[0], UuidScalarType)
    assert isinstance(scalars_a[0], UuidScalarType)
    for s, val in zip(scalars_a, data):
        assert isinstance(s, pa.ExtensionScalar)
        assert s.is_valid == (val is not None)
        assert s.type == ty1
        if val is not None:
            assert s.value == pa.scalar(val, storage.type)
            assert s.as_py() == UUID(bytes=val)
        else:
            assert s.value is None
    scalars_b = list(b)
    assert len(scalars_b) == 4
    for sa, sb in zip(scalars_a, scalars_b):
        assert isinstance(sb, pa.ExtensionScalar)
        assert sa.is_valid == sb.is_valid
        if sa.as_py() is None:
            assert sa.as_py() == sb.as_py()
        else:
            assert sa.as_py().bytes == sb.as_py()
        assert sa != sb
    scalars_c = list(c)
    assert len(scalars_c) == 4
    for s, val in zip(scalars_c, data):
        assert isinstance(s, pa.ExtensionScalar)
        assert s.is_valid == (val is not None)
        assert s.type == ty3
        if val is not None:
            assert s.value == pa.scalar(val, storage.type)
            assert s.as_py() == val
        else:
            assert s.value is None
    assert a.to_pylist() == [UUID(bytes=x) if x else None for x in data]