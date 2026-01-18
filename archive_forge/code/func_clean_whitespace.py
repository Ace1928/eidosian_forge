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
def clean_whitespace(string, compact=False):
    """Return string with compressed whitespace."""
    for a, b in (('\r\n', '\n'), ('\r', '\n'), ('\n\n', '\n'), ('\t', ' '), ('  ', ' ')):
        string = string.replace(a, b)
    if compact:
        for a, b in (('\n', ' '), ('[ ', '['), ('  ', ' '), ('  ', ' '), ('  ', ' ')):
            string = string.replace(a, b)
    return string.strip()