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
def test_create_local_intermediate_dirs():
    contents = b'meow'
    with _get_temp_local_path() as path:
        dirpath = bf.dirname(path)
        with chdir(dirpath):
            for filepath in [bf.join(dirpath, 'dirname', 'file.name'), bf.join('..', bf.basename(dirpath), 'file.name'), './file.name', 'file.name']:
                with bf.BlobFile(filepath, 'wb') as f:
                    f.write(contents)