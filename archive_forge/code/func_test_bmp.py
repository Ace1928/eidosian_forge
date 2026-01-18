from os import PathLike
import warnings
def test_bmp(h, f):
    if h.startswith(b'BM'):
        return 'bmp'