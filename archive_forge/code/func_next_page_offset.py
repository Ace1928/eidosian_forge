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
@property
def next_page_offset(self):
    """Return offset where offset to a new page can be stored."""
    if not self.complete:
        self._seek(-1)
    return self._nextpageoffset