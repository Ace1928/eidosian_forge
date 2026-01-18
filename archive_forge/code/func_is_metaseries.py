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
def is_metaseries(self):
    """Page contains MDS MetaSeries metadata in ImageDescription tag."""
    if self.index > 1 or self.software != 'MetaSeries':
        return False
    d = self.description
    return d.startswith('<MetaData>') and d.endswith('</MetaData>')