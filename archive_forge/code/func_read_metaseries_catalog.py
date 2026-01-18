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
def read_metaseries_catalog(fh):
    """Read MetaSeries non-TIFF hint catalog from file.

    Raise ValueError if the file does not contain a valid hint catalog.

    """
    raise NotImplementedError()