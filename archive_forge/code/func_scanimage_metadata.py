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
@lazyattr
def scanimage_metadata(self):
    """Return ScanImage non-varying frame and ROI metadata as dict."""
    if not self.is_scanimage:
        return
    result = {}
    try:
        framedata, roidata = read_scanimage_metadata(self._fh)
        result['FrameData'] = framedata
        result.update(roidata)
    except ValueError:
        pass
    try:
        result['Description'] = scanimage_description_metadata(self.pages[0].description)
    except Exception as e:
        warnings.warn('scanimage_description_metadata failed: %s' % e)
    return result