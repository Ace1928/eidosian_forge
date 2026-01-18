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
def test_concurrent_write_gcs():
    with _get_temp_gcs_path() as path:
        outer_contents = b'miso' * (2 ** 20 + 1)
        inner_contents = b'momo' * (2 ** 20 + 1)
        with bf.BlobFile(path, 'wb', streaming=True) as f:
            f.write(outer_contents)
            with bf.BlobFile(path, 'wb', streaming=True) as f:
                f.write(inner_contents)
        with bf.BlobFile(path, 'rb') as f:
            assert f.read() == outer_contents