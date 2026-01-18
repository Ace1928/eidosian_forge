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
def metaseries_description_metadata(description):
    """Return metatata from MetaSeries image description as dict."""
    if not description.startswith('<MetaData>'):
        raise ValueError('invalid MetaSeries image description')
    from xml.etree import cElementTree as etree
    root = etree.fromstring(description)
    types = {'float': float, 'int': int, 'bool': lambda x: asbool(x, 'on', 'off')}

    def parse(root, result):
        for child in root:
            attrib = child.attrib
            if not attrib:
                result[child.tag] = parse(child, {})
                continue
            if 'id' in attrib:
                i = attrib['id']
                t = attrib['type']
                v = attrib['value']
                if t in types:
                    result[i] = types[t](v)
                else:
                    result[i] = v
        return result
    adict = parse(root, {})
    if 'Description' in adict:
        adict['Description'] = adict['Description'].replace('&#13;&#10;', '\n')
    return adict