import gzip
import io
import struct
def snappy_decode(payload):
    if not has_snappy():
        raise NotImplementedError('Snappy codec is not available')
    if _detect_xerial_stream(payload):
        out = io.BytesIO()
        byt = payload[16:]
        length = len(byt)
        cursor = 0
        while cursor < length:
            block_size = struct.unpack_from('!i', byt[cursor:])[0]
            cursor += 4
            end = cursor + block_size
            out.write(cramjam.snappy.decompress_raw(byt[cursor:end]))
            cursor = end
        out.seek(0)
        return out.read()
    else:
        return bytes(cramjam.snappy.decompress_raw(payload))