import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
def test_field_id_metadata():
    field_id = b'PARQUET:field_id'
    inner = pa.field('inner', pa.int32(), metadata={field_id: b'100'})
    middle = pa.field('middle', pa.struct([inner]), metadata={field_id: b'101'})
    fields = [pa.field('basic', pa.int32(), metadata={b'other': b'abc', field_id: b'1'}), pa.field('list', pa.list_(pa.field('list-inner', pa.int32(), metadata={field_id: b'10'})), metadata={field_id: b'11'}), pa.field('struct', pa.struct([middle]), metadata={field_id: b'102'}), pa.field('no-metadata', pa.int32()), pa.field('non-integral-field-id', pa.int32(), metadata={field_id: b'xyz'}), pa.field('negative-field-id', pa.int32(), metadata={field_id: b'-1000'})]
    arrs = [[] for _ in fields]
    table = pa.table(arrs, schema=pa.schema(fields))
    bio = pa.BufferOutputStream()
    pq.write_table(table, bio)
    contents = bio.getvalue()
    pf = pq.ParquetFile(pa.BufferReader(contents))
    schema = pf.schema_arrow
    assert schema[0].metadata[field_id] == b'1'
    assert schema[0].metadata[b'other'] == b'abc'
    list_field = schema[1]
    assert list_field.metadata[field_id] == b'11'
    list_item_field = list_field.type.value_field
    assert list_item_field.metadata[field_id] == b'10'
    struct_field = schema[2]
    assert struct_field.metadata[field_id] == b'102'
    struct_middle_field = struct_field.type[0]
    assert struct_middle_field.metadata[field_id] == b'101'
    struct_inner_field = struct_middle_field.type[0]
    assert struct_inner_field.metadata[field_id] == b'100'
    assert schema[3].metadata is None
    assert schema[4].metadata[field_id] == b'xyz'
    assert schema[5].metadata[field_id] == b'-1000'