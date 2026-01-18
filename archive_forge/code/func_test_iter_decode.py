from __future__ import unicode_literals
from . import (lookup, LABELS, decode, encode, iter_decode, iter_encode,
def test_iter_decode():

    def iter_decode_to_string(input, fallback_encoding):
        output, _encoding = iter_decode(input, fallback_encoding)
        return ''.join(output)
    assert iter_decode_to_string([], 'latin1') == ''
    assert iter_decode_to_string([b''], 'latin1') == ''
    assert iter_decode_to_string([b'\xe9'], 'latin1') == 'é'
    assert iter_decode_to_string([b'hello'], 'latin1') == 'hello'
    assert iter_decode_to_string([b'he', b'llo'], 'latin1') == 'hello'
    assert iter_decode_to_string([b'hell', b'o'], 'latin1') == 'hello'
    assert iter_decode_to_string([b'\xc3\xa9'], 'latin1') == 'Ã©'
    assert iter_decode_to_string([b'\xef\xbb\xbf\xc3\xa9'], 'latin1') == 'é'
    assert iter_decode_to_string([b'\xef\xbb\xbf', b'\xc3', b'\xa9'], 'latin1') == 'é'
    assert iter_decode_to_string([b'\xef\xbb\xbf', b'a', b'\xc3'], 'latin1') == 'a�'
    assert iter_decode_to_string([b'', b'\xef', b'', b'', b'\xbb\xbf\xc3', b'\xa9'], 'latin1') == 'é'
    assert iter_decode_to_string([b'\xef\xbb\xbf'], 'latin1') == ''
    assert iter_decode_to_string([b'\xef\xbb'], 'latin1') == 'ï»'
    assert iter_decode_to_string([b'\xfe\xff\x00\xe9'], 'latin1') == 'é'
    assert iter_decode_to_string([b'\xff\xfe\xe9\x00'], 'latin1') == 'é'
    assert iter_decode_to_string([b'', b'\xff', b'', b'', b'\xfe\xe9', b'\x00'], 'latin1') == 'é'
    assert iter_decode_to_string([b'', b'h\xe9', b'llo'], 'x-user-defined') == 'h\uf7e9llo'