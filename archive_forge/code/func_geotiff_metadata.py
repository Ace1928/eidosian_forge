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
def geotiff_metadata(self):
    """Return GeoTIFF metadata from first page as dict."""
    if not self.is_geotiff:
        return
    return self.pages[0].geotiff_tags