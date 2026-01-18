import datetime
import warnings
import random
import string
import tempfile
import os
import contextlib
import json
import urllib.request
import hashlib
import time
import subprocess as sp
import multiprocessing as mp
import platform
import pickle
import zipfile
import re
import av
import pytest
from tensorflow.io import gfile
import imageio
import numpy as np
import blobfile as bf
from blobfile import _ops as ops, _azure as azure, _common as common
@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('streaming', [True, False])
@pytest.mark.parametrize('ctx', [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path])
def test_more_read_write(binary, streaming, ctx):
    rng = np.random.RandomState(0)
    with ctx() as path:
        if binary:
            read_mode = 'rb'
            write_mode = 'wb'
        else:
            read_mode = 'r'
            write_mode = 'w'
        with bf.BlobFile(path, write_mode, streaming=streaming) as w:
            pass
        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            assert len(r.read()) == 0
        contents = b'meow!'
        if not binary:
            contents = contents.decode('utf8')
        with bf.BlobFile(path, write_mode, streaming=streaming) as w:
            w.write(contents)
        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            assert r.read(1) == contents[:1]
            assert r.read() == contents[1:]
            assert len(r.read()) == 0
        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            for i in range(len(contents)):
                assert r.read(1) == contents[i:i + 1]
            assert len(r.read()) == 0
            assert len(r.read()) == 0
        contents = b'meow!\n\nmew!\n'
        lines = [b'meow!\n', b'\n', b'mew!\n']
        if not binary:
            contents = contents.decode('utf8')
            lines = [line.decode('utf8') for line in lines]
        with bf.BlobFile(path, write_mode, streaming=streaming) as w:
            w.write(contents)
        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            assert r.readlines() == lines
        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            assert [line for line in r] == lines
        if binary:
            for size in [2 * 2 ** 20, 12345678]:
                contents = rng.randint(0, 256, size=size, dtype=np.uint8).tobytes()
                with bf.BlobFile(path, write_mode, streaming=streaming) as w:
                    w.write(contents)
                with bf.BlobFile(path, read_mode, streaming=streaming) as r:
                    size = rng.randint(0, 1000000)
                    buf = b''
                    while True:
                        b = r.read(size)
                        if b == b'':
                            break
                        buf += b
                    assert buf == contents
        else:
            obj = {'a': 1}
            with bf.BlobFile(path, write_mode, streaming=streaming) as w:
                json.dump(obj, w)
            with bf.BlobFile(path, read_mode, streaming=streaming) as r:
                assert json.load(r) == obj