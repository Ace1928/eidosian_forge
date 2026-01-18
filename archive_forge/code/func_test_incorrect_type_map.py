from srsly.msgpack import packb, Unpacker, OutOfData
def test_incorrect_type_map():
    unpacker = Unpacker()
    unpacker.feed(packb(1))
    try:
        unpacker.read_map_header()
        assert 0, 'should raise exception'
    except UnexpectedTypeException:
        assert 1, 'okay'