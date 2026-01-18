import io
from srsly import msgpack
def test_exceeding_unpacker_read_size():
    dumpf = io.BytesIO()
    packer = msgpack.Packer()
    NUMBER_OF_STRINGS = 6
    read_size = 16
    for idx in range(NUMBER_OF_STRINGS):
        data = gen_binary_data(idx)
        dumpf.write(packer.pack(data))
    f = io.BytesIO(dumpf.getvalue())
    dumpf.close()
    unpacker = msgpack.Unpacker(f, read_size=read_size, use_list=1)
    read_count = 0
    for idx, o in enumerate(unpacker):
        assert type(o) == bytes
        assert o == gen_binary_data(idx)
        read_count += 1
    assert read_count == NUMBER_OF_STRINGS