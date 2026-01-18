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
def json_description_metadata(description):
    """Return metatata from JSON formatted image description as dict.

    Raise ValuError if description is of unknown format.

    >>> description = '{"shape": [256, 256, 3], "axes": "YXS"}'
    >>> json_description_metadata(description)  # doctest: +SKIP
    {'shape': [256, 256, 3], 'axes': 'YXS'}
    >>> json_description_metadata('shape=(256, 256, 3)')
    {'shape': (256, 256, 3)}

    """
    if description[:6] == 'shape=':
        shape = tuple((int(i) for i in description[7:-1].split(',')))
        return dict(shape=shape)
    if description[:1] == '{' and description[-1:] == '}':
        return json.loads(description)
    raise ValueError('invalid JSON image description', description)