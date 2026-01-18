import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET
@pytest.mark.parametrize('pa_type,jvm_spec', [(pa.null(), '{"name":"null"}'), (pa.bool_(), '{"name":"bool"}'), (pa.int8(), '{"name":"int","bitWidth":8,"isSigned":true}'), (pa.int16(), '{"name":"int","bitWidth":16,"isSigned":true}'), (pa.int32(), '{"name":"int","bitWidth":32,"isSigned":true}'), (pa.int64(), '{"name":"int","bitWidth":64,"isSigned":true}'), (pa.uint8(), '{"name":"int","bitWidth":8,"isSigned":false}'), (pa.uint16(), '{"name":"int","bitWidth":16,"isSigned":false}'), (pa.uint32(), '{"name":"int","bitWidth":32,"isSigned":false}'), (pa.uint64(), '{"name":"int","bitWidth":64,"isSigned":false}'), (pa.float16(), '{"name":"floatingpoint","precision":"HALF"}'), (pa.float32(), '{"name":"floatingpoint","precision":"SINGLE"}'), (pa.float64(), '{"name":"floatingpoint","precision":"DOUBLE"}'), (pa.time32('s'), '{"name":"time","unit":"SECOND","bitWidth":32}'), (pa.time32('ms'), '{"name":"time","unit":"MILLISECOND","bitWidth":32}'), (pa.time64('us'), '{"name":"time","unit":"MICROSECOND","bitWidth":64}'), (pa.time64('ns'), '{"name":"time","unit":"NANOSECOND","bitWidth":64}'), (pa.timestamp('s'), '{"name":"timestamp","unit":"SECOND","timezone":null}'), (pa.timestamp('ms'), '{"name":"timestamp","unit":"MILLISECOND","timezone":null}'), (pa.timestamp('us'), '{"name":"timestamp","unit":"MICROSECOND","timezone":null}'), (pa.timestamp('ns'), '{"name":"timestamp","unit":"NANOSECOND","timezone":null}'), (pa.timestamp('ns', tz='UTC'), '{"name":"timestamp","unit":"NANOSECOND","timezone":"UTC"}'), (pa.timestamp('ns', tz='Europe/Paris'), '{"name":"timestamp","unit":"NANOSECOND","timezone":"Europe/Paris"}'), (pa.date32(), '{"name":"date","unit":"DAY"}'), (pa.date64(), '{"name":"date","unit":"MILLISECOND"}'), (pa.decimal128(19, 4), '{"name":"decimal","precision":19,"scale":4}'), (pa.string(), '{"name":"utf8"}'), (pa.binary(), '{"name":"binary"}'), (pa.binary(10), '{"name":"fixedsizebinary","byteWidth":10}')])
@pytest.mark.parametrize('nullable', [True, False])
def test_jvm_types(root_allocator, pa_type, jvm_spec, nullable):
    if pa_type == pa.null() and (not nullable):
        return
    spec = {'name': 'field_name', 'nullable': nullable, 'type': json.loads(jvm_spec), 'children': []}
    jvm_field = _jvm_field(json.dumps(spec))
    result = pa_jvm.field(jvm_field)
    expected_field = pa.field('field_name', pa_type, nullable=nullable)
    assert result == expected_field
    jvm_schema = _jvm_schema(json.dumps(spec))
    result = pa_jvm.schema(jvm_schema)
    assert result == pa.schema([expected_field])
    jvm_schema = _jvm_schema(json.dumps(spec), {'meta': 'data'})
    result = pa_jvm.schema(jvm_schema)
    assert result == pa.schema([expected_field], {'meta': 'data'})
    spec['metadata'] = [{'key': 'field meta', 'value': 'field data'}]
    jvm_schema = _jvm_schema(json.dumps(spec))
    result = pa_jvm.schema(jvm_schema)
    expected_field = expected_field.with_metadata({'field meta': 'field data'})
    assert result == pa.schema([expected_field])