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
def test_more_exists():
    testcases = [(AZURE_INVALID_CONTAINER, False), (AZURE_INVALID_CONTAINER + '/', False), (AZURE_INVALID_CONTAINER + '//', False), (AZURE_INVALID_CONTAINER + '/invalid.file', False), (GCS_INVALID_BUCKET, False), (GCS_INVALID_BUCKET + '/', False), (GCS_INVALID_BUCKET + '//', False), (GCS_INVALID_BUCKET + '/invalid.file', False), (AZURE_INVALID_CONTAINER_NO_ACCOUNT, False), (AZURE_INVALID_CONTAINER_NO_ACCOUNT + '/', False), (AZURE_INVALID_CONTAINER_NO_ACCOUNT + '//', False), (AZURE_INVALID_CONTAINER_NO_ACCOUNT + '/invalid.file', False), (AZURE_VALID_CONTAINER, True), (AZURE_VALID_CONTAINER + '/', True), (AZURE_VALID_CONTAINER + '//', False), (AZURE_VALID_CONTAINER + '/invalid.file', False), (GCS_VALID_BUCKET, True), (GCS_VALID_BUCKET + '/', True), (GCS_VALID_BUCKET + '//', False), (GCS_VALID_BUCKET + '/invalid.file', False), ('/does-not-exist', False), ('/', True)]
    for path, should_exist in testcases:
        assert bf.exists(path) == should_exist