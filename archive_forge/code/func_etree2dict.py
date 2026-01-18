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
def etree2dict(t):
    key = t.tag
    if sanitize:
        key = key.rsplit('}', 1)[-1]
    d = {key: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = collections.defaultdict(list)
        for dc in map(etree2dict, children):
            for k, v in dc.items():
                dd[k].append(astype(v))
        d = {key: {k: astype(v[0]) if len(v) == 1 else astype(v) for k, v in dd.items()}}
    if t.attrib:
        d[key].update(((at + k, astype(v)) for k, v in t.attrib.items()))
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[key][tx + 'value'] = astype(text)
        else:
            d[key] = astype(text)
    return d