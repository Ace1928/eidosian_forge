from __future__ import unicode_literals
from . import (lookup, LABELS, decode, encode, iter_decode, iter_encode,
def test_iter_encode():
    assert b''.join(iter_encode([], 'latin1')) == b''
    assert b''.join(iter_encode([''], 'latin1')) == b''
    assert b''.join(iter_encode(['é'], 'latin1')) == b'\xe9'
    assert b''.join(iter_encode(['', 'é', '', ''], 'latin1')) == b'\xe9'
    assert b''.join(iter_encode(['', 'é', '', ''], 'utf-16')) == b'\xe9\x00'
    assert b''.join(iter_encode(['', 'é', '', ''], 'utf-16le')) == b'\xe9\x00'
    assert b''.join(iter_encode(['', 'é', '', ''], 'utf-16be')) == b'\x00\xe9'
    assert b''.join(iter_encode(['', 'h\uf7e9', '', 'llo'], 'x-user-defined')) == b'h\xe9llo'