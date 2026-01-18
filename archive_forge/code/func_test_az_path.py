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
def test_az_path():
    contents = b'meow!\npurr\n'
    with _get_temp_as_path() as path:
        path = _convert_https_to_az(path)
        path = bf.join(path, 'a folder', 'a.file')
        path = _convert_https_to_az(path)
        bf.makedirs(_convert_https_to_az(bf.dirname(path)))
        with bf.BlobFile(path, 'wb') as w:
            w.write(contents)
        with bf.BlobFile(path, 'rb') as r:
            assert r.read() == contents
        with bf.BlobFile(path, 'rb') as r:
            lines = list(r)
            assert b''.join(lines) == contents