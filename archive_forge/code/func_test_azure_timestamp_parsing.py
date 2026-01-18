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
def test_azure_timestamp_parsing():
    timestamp = 'Sun, 27 Sep 2009 18:41:57 GMT'

    def ref_parse_timestamp(text: str) -> float:
        return datetime.datetime.strptime(text.replace('GMT', 'Z'), '%a, %d %b %Y %H:%M:%S %z').timestamp()
    assert ref_parse_timestamp(timestamp) == azure._parse_timestamp(timestamp)