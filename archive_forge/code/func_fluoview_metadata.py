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
def fluoview_metadata(self):
    """Return consolidated FluoView metadata as dict."""
    if not self.is_fluoview:
        return
    result = {}
    page = self.pages[0]
    result.update(page.tags['MM_Header'].value)
    result['Stamp'] = page.tags['MM_Stamp'].value
    return result