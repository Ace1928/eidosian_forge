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
def shaped_metadata(self):
    """Return Tifffile metadata from JSON descriptions as dicts."""
    if not self.is_shaped:
        return
    return tuple((json_description_metadata(s.pages[0].is_shaped) for s in self.series if s.stype.lower() == 'shaped'))