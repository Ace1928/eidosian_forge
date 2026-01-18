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
def test_composite_objects():
    with _get_temp_gcs_path() as remote_path:
        with _get_temp_local_path() as local_path:
            contents = b'0' * 2 * 2 ** 20
            with open(local_path, 'wb') as f:
                f.write(contents)

            def create_composite_file():
                sp.run(['gsutil', '-o', 'GSUtil:parallel_composite_upload_threshold=1M', 'cp', local_path, remote_path], check=True)
            local_md5 = hashlib.md5(contents).hexdigest()
            create_composite_file()
            assert bf.stat(remote_path).md5 is None
            assert local_md5 == bf.md5(remote_path)
            assert bf.stat(remote_path).md5 == local_md5
            assert local_md5 == bf.md5(remote_path)
            bf.remove(remote_path)
            create_composite_file()
            assert bf.stat(remote_path).md5 is None
            with tempfile.TemporaryDirectory() as tmpdir:
                with bf.BlobFile(remote_path, 'rb', cache_dir=tmpdir, streaming=False) as f:
                    assert f.read() == contents
            assert bf.stat(remote_path).md5 == local_md5