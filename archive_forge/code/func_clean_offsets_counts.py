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
def clean_offsets_counts(offsets, counts):
    """Return cleaned offsets and byte counts.

    Remove zero offsets and counts. Use to sanitize _offsets and _bytecounts
    tag values for strips or tiles.

    """
    offsets = list(offsets)
    counts = list(counts)
    assert len(offsets) == len(counts)
    j = 0
    for i, (o, b) in enumerate(zip(offsets, counts)):
        if o > 0 and b > 0:
            if i > j:
                offsets[j] = o
                counts[j] = b
            j += 1
        elif b > 0 and o <= 0:
            raise ValueError('invalid offset')
        else:
            warnings.warn('empty byte count')
    if j == 0:
        j = 1
    return (offsets[:j], counts[:j])