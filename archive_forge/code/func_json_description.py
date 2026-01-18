from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def json_description(shape, **metadata):
    """Return JSON image description from data shape and other meta data.

    Return UTF-8 encoded JSON.

    >>> json_description((256, 256, 3), axes='YXS')  # doctest: +SKIP
    b'{"shape": [256, 256, 3], "axes": "YXS"}'

    """
    metadata.update(shape=shape)
    return json.dumps(metadata)