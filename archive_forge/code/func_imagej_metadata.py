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
def imagej_metadata(self):
    """Return consolidated ImageJ metadata as dict."""
    if not self.is_imagej:
        return
    page = self.pages[0]
    result = imagej_description_metadata(page.is_imagej)
    if 'IJMetadata' in page.tags:
        try:
            result.update(page.tags['IJMetadata'].value)
        except Exception:
            pass
    return result