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
def test_azure_maybe_update_md5(ctx):
    contents = b'meow!'
    meow_hash = hashlib.md5(contents).hexdigest()
    alternative_contents = b'purr'
    purr_hash = hashlib.md5(alternative_contents).hexdigest()
    with ctx() as path:
        _write_contents(path, contents)
        st = azure.maybe_stat(ops.default_context._conf, path)
        assert azure.maybe_update_md5(ops.default_context._conf, path, st.version, meow_hash)
        _write_contents(path, alternative_contents)
        assert not azure.maybe_update_md5(ops.default_context._conf, path, st.version, meow_hash)
        st = azure.maybe_stat(ops.default_context._conf, path)
        assert st.md5 == purr_hash
        bf.remove(path)
        assert not azure.maybe_update_md5(ops.default_context._conf, path, st.version, meow_hash)