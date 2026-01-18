import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def extractRAWJSON(self, cb=None, sort=False) -> str:
    """Return 'extractRAWDICT' converted to JSON format."""
    import base64
    import json
    val = self._textpage_dict(raw=True)

    class b64encode(json.JSONEncoder):

        def default(self, s):
            if type(s) in (bytes, bytearray):
                return base64.b64encode(s).decode()
    if cb is not None:
        val['width'] = cb.width
        val['height'] = cb.height
    if sort is True:
        blocks = val['blocks']
        blocks.sort(key=lambda b: (b['bbox'][3], b['bbox'][0]))
        val['blocks'] = blocks
    val = json.dumps(val, separators=(',', ':'), cls=b64encode, indent=1)
    return val