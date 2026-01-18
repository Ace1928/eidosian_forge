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
@pytest.mark.parametrize('base_path', [AZURE_INVALID_CONTAINER_NO_ACCOUNT, AZURE_INVALID_CONTAINER, GCS_INVALID_BUCKET])
def test_invalid_paths(base_path):
    for suffix in ['', '/', '//', '/invalid.file', '/invalid/dir/']:
        path = base_path + suffix
        print(path)
        if path.endswith('/'):
            expected_error = IsADirectoryError
        else:
            expected_error = FileNotFoundError
        list(bf.glob(path))
        if suffix == '':
            for pattern in ['*', '**']:
                try:
                    list(bf.glob(path + pattern))
                except bf.Error as e:
                    assert 'Wildcards cannot be used' in e.message
        else:
            for pattern in ['*', '**']:
                list(bf.glob(path + pattern))
        with pytest.raises(FileNotFoundError):
            list(bf.listdir(path))
        assert not bf.exists(path)
        assert not bf.isdir(path)
        with pytest.raises(expected_error):
            bf.remove(path)
        if suffix in ('', '/'):
            try:
                bf.rmdir(path)
            except bf.Error as e:
                assert 'Cannot delete bucket' in e.message
        else:
            bf.rmdir(path)
        with pytest.raises(NotADirectoryError):
            bf.rmtree(path)
        with pytest.raises(FileNotFoundError):
            bf.stat(path)
        if base_path == AZURE_INVALID_CONTAINER_NO_ACCOUNT:
            with pytest.raises(bf.Error):
                bf.get_url(path)
        else:
            bf.get_url(path)
        with pytest.raises(FileNotFoundError):
            bf.md5(path)
        with pytest.raises(bf.Error):
            bf.makedirs(path)
        list(bf.walk(path))
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, 'test.txt')
            with pytest.raises(expected_error):
                bf.copy(path, local_path)
            with open(local_path, 'w') as f:
                f.write('meow')
            with pytest.raises(expected_error):
                bf.copy(local_path, path)
        for streaming in [False, True]:
            with pytest.raises(expected_error):
                with bf.BlobFile(path, 'rb', streaming=streaming) as f:
                    f.read()
            with pytest.raises(expected_error):
                with bf.BlobFile(path, 'wb', streaming=streaming) as f:
                    f.write(b'meow')