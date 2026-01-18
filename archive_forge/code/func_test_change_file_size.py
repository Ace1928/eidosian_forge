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
@pytest.mark.parametrize('ctx', [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path])
@pytest.mark.parametrize('use_random', [False, True])
def test_change_file_size(ctx, use_random):
    chunk_size = 8 * 2 ** 20
    long_contents = b'\x00' * chunk_size * 3
    short_contents = b'\xff' * chunk_size * 2
    if use_random:
        long_contents = os.urandom(len(long_contents))
        short_contents = os.urandom(len(short_contents))
    with ctx() as path:
        with bf.BlobFile(path, 'wb') as f:
            f.write(long_contents)
        with bf.BlobFile(path, 'rb') as f:
            read_contents = f.read(chunk_size)
            with bf.BlobFile(path, 'wb') as f2:
                f2.write(short_contents)
            f.raw._f = None
            read_contents += f.read()
            assert len(f.read()) == 0
            assert read_contents == long_contents[:chunk_size] + short_contents[chunk_size:]
        with bf.BlobFile(path, 'wb') as f:
            f.write(short_contents)
        with bf.BlobFile(path, 'rb') as f:
            read_contents = f.read(chunk_size)
            with bf.BlobFile(path, 'wb') as f2:
                f2.write(long_contents)
            f.raw._f = None
            read_contents += f.read()
            assert len(f.read()) == 0
            expected = short_contents[:chunk_size] + long_contents[chunk_size:chunk_size * 2]
            if not path.startswith('gs://') and (not path.startswith('https://')):
                expected = short_contents[:chunk_size] + long_contents[chunk_size:]
            assert read_contents == expected