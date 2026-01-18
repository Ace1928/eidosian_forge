from srsly.msgpack import packb, Unpacker, OutOfData
def test_correct_type_nested_array():
    unpacker = Unpacker()
    unpacker.feed(packb({'a': ['b', 'c', 'd']}))
    try:
        unpacker.read_array_header()
        assert 0, 'should raise exception'
    except UnexpectedTypeException:
        assert 1, 'okay'