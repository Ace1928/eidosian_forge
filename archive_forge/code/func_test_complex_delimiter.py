from __future__ import annotations
from functools import partial
import pytest
from fsspec.compression import compr
from tlz import concat
from dask import compute, config
from dask.bag.text import read_text
from dask.bytes import utils
from dask.utils import filetexts
def test_complex_delimiter():
    longstr = 'abc\ndef\n123\n$$$$\ndog\ncat\nfish\n\n\r\n$$$$hello'
    with filetexts({'.test.delim.txt': longstr}):
        assert read_text('.test.delim.txt', linedelimiter='$$$$').count().compute() == 3
        assert read_text('.test.delim.txt', linedelimiter='$$$$', blocksize=2).count().compute() == 3
        vals = read_text('.test.delim.txt', linedelimiter='$$$$').compute()
        assert vals[-1] == 'hello'
        assert vals[0].endswith('$$$$')
        vals = read_text('.test.delim.txt', linedelimiter='$$$$', blocksize=2).compute()
        assert vals[-1] == 'hello'
        assert vals[0].endswith('$$$$')