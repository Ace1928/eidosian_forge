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
def test_azure_public_get_url():
    contents = urllib.request.urlopen(AZURE_PUBLIC_URL).read()
    assert contents.startswith(AZURE_PUBLIC_URL_HEADER)
    url, _ = bf.get_url(AZURE_PUBLIC_URL)
    assert urllib.request.urlopen(url).read() == contents