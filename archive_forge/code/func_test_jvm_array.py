import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET
@pytest.mark.parametrize('pa_type,py_data,jvm_type', [(pa.bool_(), [True, False, True, True], 'BitVector'), (pa.uint8(), list(range(128)), 'UInt1Vector'), (pa.uint16(), list(range(128)), 'UInt2Vector'), (pa.int32(), list(range(128)), 'IntVector'), (pa.int64(), list(range(128)), 'BigIntVector'), (pa.float32(), list(range(128)), 'Float4Vector'), (pa.float64(), list(range(128)), 'Float8Vector'), (pa.timestamp('s'), list(range(128)), 'TimeStampSecVector'), (pa.timestamp('ms'), list(range(128)), 'TimeStampMilliVector'), (pa.timestamp('us'), list(range(128)), 'TimeStampMicroVector'), (pa.timestamp('ns'), list(range(128)), 'TimeStampNanoVector'), (pa.date32(), list(range(128)), 'DateDayVector'), (pa.date64(), list(range(128)), 'DateMilliVector')])
def test_jvm_array(root_allocator, pa_type, py_data, jvm_type):
    cls = 'org.apache.arrow.vector.{}'.format(jvm_type)
    jvm_vector = jpype.JClass(cls)('vector', root_allocator)
    jvm_vector.allocateNew(len(py_data))
    for i, val in enumerate(py_data):
        if jvm_type in {'UInt1Vector', 'UInt2Vector'}:
            val = jpype.JInt(val)
        jvm_vector.setSafe(i, val)
    jvm_vector.setValueCount(len(py_data))
    py_array = pa.array(py_data, type=pa_type)
    jvm_array = pa_jvm.array(jvm_vector)
    assert py_array.equals(jvm_array)