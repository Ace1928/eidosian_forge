from os import PathLike
import warnings
def test_exr(h, f):
    if h.startswith(b'v/1\x01'):
        return 'exr'