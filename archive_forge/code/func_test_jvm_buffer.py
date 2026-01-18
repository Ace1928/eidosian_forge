import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET
def test_jvm_buffer(root_allocator):
    jvm_buffer = root_allocator.buffer(8)
    for i in range(8):
        jvm_buffer.setByte(i, 8 - i)
    orig_refcnt = jvm_buffer.refCnt()
    buf = pa_jvm.jvm_buffer(jvm_buffer)
    assert buf.to_pybytes() == b'\x08\x07\x06\x05\x04\x03\x02\x01'
    assert jvm_buffer.refCnt() == orig_refcnt + 1
    del buf
    assert jvm_buffer.refCnt() == orig_refcnt