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
def test_azure_public_container():
    for error, path in [(None, f'https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/publiccontainer/test_cat.png'), (bf.Error, f'https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/private/test_cat.png'), (FileNotFoundError, f'https://{AS_INVALID_ACCOUNT}.blob.core.windows.net/publiccontainer/test_cat.png')]:
        ctx = contextlib.nullcontext()
        if error is not None:
            ctx = pytest.raises(error)
        with ctx:
            with bf.BlobFile(path, 'rb') as f:
                contents = f.read()
                assert contents.startswith(AZURE_PUBLIC_URL_HEADER)