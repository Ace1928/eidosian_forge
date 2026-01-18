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
def ome_metadata(self):
    """Return OME XML as dict."""
    if not self.is_ome:
        return
    return xml2dict(self.pages[0].description)['OME']