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
@pytest.mark.parametrize('ctx', [_get_temp_as_path])
def test_azure_metadata(ctx):
    contents = b'meow!'
    with ctx() as path:
        with bf.BlobFile(path, 'wb') as f:
            f.write(contents)
        bf.set_mtime(path, 1)
        time.sleep(5)
        with bf.BlobFile(path, 'wb', streaming=True) as f:
            st = bf.stat(path)
        assert st.mtime == 1