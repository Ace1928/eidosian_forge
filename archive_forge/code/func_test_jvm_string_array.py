import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET
@pytest.mark.xfail(reason='from_buffers is only supported for primitive arrays yet')
def test_jvm_string_array(root_allocator):
    data = ['string', None, 'töst']
    cls = 'org.apache.arrow.vector.VarCharVector'
    jvm_vector = jpype.JClass(cls)('vector', root_allocator)
    jvm_vector.allocateNew()
    for i, string in enumerate(data):
        holder = _string_to_varchar_holder(root_allocator, 'string')
        jvm_vector.setSafe(i, holder)
        jvm_vector.setValueCount(i + 1)
    py_array = pa.array(data, type=pa.string())
    jvm_array = pa_jvm.array(jvm_vector)
    assert py_array.equals(jvm_array)