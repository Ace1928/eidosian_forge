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
@pytest.mark.slow
@pytest.mark.parametrize('ctx', [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path])
def test_large_file(ctx):
    contents = b'0' * 2 ** 32
    with ctx() as path:
        with bf.BlobFile(path, 'wb', streaming=True) as f:
            f.write(contents)
        with bf.BlobFile(path, 'rb', streaming=True) as f:
            assert contents == f.read()